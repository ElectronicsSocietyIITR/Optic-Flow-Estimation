#include "preprocess.hpp"

#include "common/xf_common.hpp"
#include "imgproc/xf_pyr_down.hpp"
#include "imgproc/xf_resize.hpp"

namespace {

using Bgr = xf::cv::Mat<XF_8UC3, kSrcMaxRows, kSrcMaxCols, XF_NPPC1, 2>;
using BgrSmall = xf::cv::Mat<XF_8UC3, kDstMaxRows, kDstMaxCols, XF_NPPC1, 2>;
using Gray0 = xf::cv::Mat<XF_8UC1, kDstMaxRows, kDstMaxCols, XF_NPPC1, 2>;
using Gray1 = xf::cv::Mat<XF_8UC1, kDstMaxRows / 2, kDstMaxCols / 2, XF_NPPC1, 2>;
using Gray2 = xf::cv::Mat<XF_8UC1, kDstMaxRows / 4, kDstMaxCols / 4, XF_NPPC1, 2>;

void readFrame(const ap_uint<32>* bgr, Bgr& src, int n) {
read:
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
    src.write(i, bgr[i].range(23, 0));
  }
}

// One pass over the resized frame: the DPU tensor, gray level 0 to memory, and gray level 0 again
// as a stream for the pyramid.
void tensorAndGray(BgrSmall& small, ap_int<8>* tensor, ap_uint<8>* gray0, Gray0& g0, int n, int input_shift) {
split:
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
    const ap_uint<24> p = small.read(i);
    const ap_uint<8> b = p.range(7, 0), g = p.range(15, 8), r = p.range(23, 16);
    // INT8 tensor, RGB order: round(v * 2^shift / 255), at most 127.
    const ap_uint<8> rgb[3] = {r, g, b};
    for (int c = 0; c < 3; ++c) {
      const ap_uint<16> q = (ap_uint<16>(rgb[c]) * (ap_uint<16>(1) << input_shift) * 2 + 255) / 510;  // round half up
      tensor[3 * i + c] = q > 127 ? ap_int<8>(127) : ap_int<8>(q);
    }
    // OpenCV's 8-bit BGR->gray: (4899 R + 9617 G + 1868 B + 2^13) >> 14.
    const ap_uint<8> y = (ap_uint<24>(4899) * r + ap_uint<24>(9617) * g + ap_uint<24>(1868) * b + 8192) >> 14;
    gray0[i] = y;
    g0.write(i, y);
  }
}

template <class M>
void writeAndPass(M& in, ap_uint<8>* out, M& again, int n) {
copy:
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
    const ap_uint<8> v = in.read(i);
    out[i] = v;
    again.write(i, v);
  }
}

template <class M>
void writeOut(M& in, ap_uint<8>* out, int n) {
copy:
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
    out[i] = in.read(i);
  }
}

}  // namespace

void preprocess(const ap_uint<32>* bgr, ap_int<8>* tensor, ap_uint<8>* gray0, ap_uint<8>* gray1, ap_uint<8>* gray2, int src_rows,
                int src_cols, int dst_rows, int dst_cols, int input_shift) {
#pragma HLS INTERFACE m_axi port = bgr offset = slave bundle = gmem0 depth = 2073600
#pragma HLS INTERFACE m_axi port = tensor offset = slave bundle = gmem1 depth = 786432
#pragma HLS INTERFACE m_axi port = gray0 offset = slave bundle = gmem2 depth = 262144
#pragma HLS INTERFACE m_axi port = gray1 offset = slave bundle = gmem3 depth = 65536
#pragma HLS INTERFACE m_axi port = gray2 offset = slave bundle = gmem4 depth = 16384
#pragma HLS INTERFACE s_axilite port = src_rows
#pragma HLS INTERFACE s_axilite port = src_cols
#pragma HLS INTERFACE s_axilite port = dst_rows
#pragma HLS INTERFACE s_axilite port = dst_cols
#pragma HLS INTERFACE s_axilite port = input_shift
#pragma HLS INTERFACE s_axilite port = return
  const int r1 = (dst_rows + 1) / 2, c1 = (dst_cols + 1) / 2, r2 = (r1 + 1) / 2, c2 = (c1 + 1) / 2;
  static Bgr src(src_rows, src_cols);
  static BgrSmall small(dst_rows, dst_cols);
  static Gray0 g0(dst_rows, dst_cols);
  static Gray0 g0b(dst_rows, dst_cols);
  static Gray1 g1(r1, c1), g1b(r1, c1);
  static Gray2 g2(r2, c2);
  src.rows = src_rows; src.cols = src_cols;
  small.rows = dst_rows; small.cols = dst_cols;
  g0.rows = g0b.rows = dst_rows; g0.cols = g0b.cols = dst_cols;
  g1.rows = g1b.rows = r1; g1.cols = g1b.cols = c1;
  g2.rows = r2; g2.cols = c2;
#pragma HLS DATAFLOW
  readFrame(bgr, src, src_rows * src_cols);
  xf::cv::resize<XF_INTERPOLATION_BILINEAR, XF_8UC3, kSrcMaxRows, kSrcMaxCols, kDstMaxRows, kDstMaxCols, XF_NPPC1, false, 4>(src, small);
  tensorAndGray(small, tensor, gray0, g0, dst_rows * dst_cols, input_shift);
  xf::cv::pyrDown<XF_8UC1, kDstMaxRows, kDstMaxCols, XF_NPPC1, false>(g0, g1);
  writeAndPass(g1, gray1, g1b, r1 * c1);
  xf::cv::pyrDown<XF_8UC1, kDstMaxRows / 2, kDstMaxCols / 2, XF_NPPC1, false>(g1b, g2);
  writeOut(g2, gray2, r2 * c2);
}
