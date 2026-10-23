#include "preprocess.hpp"

#include "common/xf_common.hpp"
#include "imgproc/xf_pyr_down.hpp"
#include "imgproc/xf_resize.hpp"

namespace {

using Bgr = xf::cv::Mat<XF_8UC3, kSrcMaxRows, kSrcMaxCols, XF_NPPC1, 2>;
using BgrSmall = xf::cv::Mat<XF_8UC3, kDstMaxRows, kDstMaxCols, XF_NPPC1, 2>;
template <int D>
using Gray = xf::cv::Mat<XF_8UC1, kDstMaxRows / D, kDstMaxCols / D, XF_NPPC1, 2>;

void readFrame(const ap_uint<32>* bgr, Bgr& src, int n) {
read:
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
    src.write(i, bgr[i].range(23, 0));
  }
}

// One pass over the resized frame: the DPU tensor, gray level 0 to memory, and level 0 again as a
// stream for the pyramid.
void tensorAndGray(BgrSmall& small, ap_int<8>* tensor, ap_uint<8>* level0, Gray<1>& g0, int n, int input_shift) {
split:
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
    const ap_uint<24> p = small.read(i);
    const ap_uint<8> b = p.range(7, 0), g = p.range(15, 8), r = p.range(23, 16);
    const ap_uint<8> rgb[3] = {r, g, b};
    for (int c = 0; c < 3; ++c) {  // round half up of v * 2^shift / 255, at most 127
      const ap_uint<16> q = (ap_uint<16>(rgb[c]) * (ap_uint<16>(1) << input_shift) * 2 + 255) / 510;
      tensor[3 * i + c] = q > 127 ? ap_int<8>(127) : ap_int<8>(q);
    }
    const ap_uint<8> y = (ap_uint<24>(4899) * r + ap_uint<24>(9617) * g + ap_uint<24>(1868) * b + 8192) >> 14;
    level0[i] = y;
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

void preprocess(const ap_uint<32>* bgr, ap_int<8>* tensor, ap_uint<8>* pyr, int src_rows, int src_cols, int dst_rows, int dst_cols,
                int input_shift) {
#pragma HLS INTERFACE m_axi port = bgr offset = slave bundle = gmem0 depth = 2073600
#pragma HLS INTERFACE m_axi port = tensor offset = slave bundle = gmem1 depth = 786432
#pragma HLS INTERFACE m_axi port = pyr offset = slave bundle = gmem2 depth = 349696
#pragma HLS INTERFACE s_axilite port = src_rows
#pragma HLS INTERFACE s_axilite port = src_cols
#pragma HLS INTERFACE s_axilite port = dst_rows
#pragma HLS INTERFACE s_axilite port = dst_cols
#pragma HLS INTERFACE s_axilite port = input_shift
#pragma HLS INTERFACE s_axilite port = return
  int rows[kPyrLevels], cols[kPyrLevels], off[kPyrLevels];
  for (int l = 0; l < kPyrLevels; ++l) {
    rows[l] = pyrRows(dst_rows, l);
    cols[l] = pyrCols(dst_cols, l);
    off[l] = pyrOffset(dst_rows, dst_cols, l);
  }
  static Bgr src;
  static BgrSmall small;
  static Gray<1> g0;
  static Gray<2> g1, g1b;
  static Gray<4> g2, g2b;
  static Gray<8> g3, g3b;
  static Gray<16> g4;
  src.rows = src_rows; src.cols = src_cols;
  small.rows = dst_rows; small.cols = dst_cols;
  g0.rows = rows[0]; g0.cols = cols[0];
  g1.rows = g1b.rows = rows[1]; g1.cols = g1b.cols = cols[1];
  g2.rows = g2b.rows = rows[2]; g2.cols = g2b.cols = cols[2];
  g3.rows = g3b.rows = rows[3]; g3.cols = g3b.cols = cols[3];
  g4.rows = rows[4]; g4.cols = cols[4];
  src.size = src_rows * src_cols;
  small.size = dst_rows * dst_cols;
  g0.size = rows[0] * cols[0];
  g1.size = g1b.size = rows[1] * cols[1];
  g2.size = g2b.size = rows[2] * cols[2];
  g3.size = g3b.size = rows[3] * cols[3];
  g4.size = rows[4] * cols[4];
#pragma HLS DATAFLOW
  readFrame(bgr, src, src_rows * src_cols);
  xf::cv::resize<XF_INTERPOLATION_BILINEAR, XF_8UC3, kSrcMaxRows, kSrcMaxCols, kDstMaxRows, kDstMaxCols, XF_NPPC1, false, 4>(src, small);
  tensorAndGray(small, tensor, pyr + off[0], g0, rows[0] * cols[0], input_shift);
  xf::cv::pyrDown<XF_8UC1, kDstMaxRows, kDstMaxCols, XF_NPPC1, false>(g0, g1);
  writeAndPass(g1, pyr + off[1], g1b, rows[1] * cols[1]);
  xf::cv::pyrDown<XF_8UC1, kDstMaxRows / 2, kDstMaxCols / 2, XF_NPPC1, false>(g1b, g2);
  writeAndPass(g2, pyr + off[2], g2b, rows[2] * cols[2]);
  xf::cv::pyrDown<XF_8UC1, kDstMaxRows / 4, kDstMaxCols / 4, XF_NPPC1, false>(g2b, g3);
  writeAndPass(g3, pyr + off[3], g3b, rows[3] * cols[3]);
  xf::cv::pyrDown<XF_8UC1, kDstMaxRows / 8, kDstMaxCols / 8, XF_NPPC1, false>(g3b, g4);
  writeOut(g4, pyr + off[4], rows[4] * cols[4]);
}
