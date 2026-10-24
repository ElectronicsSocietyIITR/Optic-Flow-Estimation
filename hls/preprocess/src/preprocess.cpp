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
void tensorAndGray(BgrSmall& small, ap_int<8>* tensor, ap_uint<8>* level0, int n, int input_shift) {
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
    level0[i] = (ap_uint<24>(4899) * r + ap_uint<24>(9617) * g + ap_uint<24>(1868) * b + 8192) >> 14;
  }
}

void readLevel(const ap_uint<8>* in, Gray<1>& m, int n) {
read:
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
    m.write(i, in[i]);
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

// The camera frame -> resize -> DPU tensor + pyramid level 0, one streaming pass.
void frameToLevel0(const ap_uint<32>* bgr, ap_int<8>* tensor, ap_uint<8>* level0, int src_rows, int src_cols, int rows, int cols,
                   int input_shift) {
  static Bgr src;
  static BgrSmall small;
  src.rows = src_rows; src.cols = src_cols; src.size = src_rows * src_cols;
  small.rows = rows; small.cols = cols; small.size = rows * cols;
#pragma HLS DATAFLOW
  readFrame(bgr, src, src_rows * src_cols);
  xf::cv::resize<XF_INTERPOLATION_BILINEAR, XF_8UC3, kSrcMaxRows, kSrcMaxCols, kDstMaxRows, kDstMaxCols, XF_NPPC1, false, 4>(src, small);
  tensorAndGray(small, tensor, level0, rows * cols, input_shift);
}

// One pyramid level from the one above it, through DDR. One pyrDown instance serves all four levels (they
// add only ~1/3 of level 0's pixels, ~0.2 ms in all): four instances cost ~13k LUTs in synthesis.
void pyramidLevel(const ap_uint<8>* in, ap_uint<8>* out, int rows, int cols, int rows2, int cols2) {
  static Gray<1> above;
  static Gray<2> below;  // pyrDown writes (rows + 1) / 2 x (cols + 1) / 2
  above.rows = rows; above.cols = cols; above.size = rows * cols;
  below.rows = rows2; below.cols = cols2; below.size = rows2 * cols2;
#pragma HLS DATAFLOW
  readLevel(in, above, rows * cols);
  xf::cv::pyrDown<XF_8UC1, kDstMaxRows, kDstMaxCols, XF_NPPC1, false>(above, below);
  writeOut(below, out, rows2 * cols2);
}

}  // namespace

void preprocess(const ap_uint<32>* bgr, ap_int<8>* tensor, ap_uint<8>* pyr, const ap_uint<8>* pyr_rd, int src_rows, int src_cols,
                int dst_rows, int dst_cols, int input_shift) {
#pragma HLS INTERFACE m_axi port = bgr offset = slave bundle = gmem0 depth = 2073600
#pragma HLS INTERFACE m_axi port = tensor offset = slave bundle = gmem1 depth = 786432
#pragma HLS INTERFACE m_axi port = pyr offset = slave bundle = gmem2 depth = 349696
#pragma HLS INTERFACE m_axi port = pyr_rd offset = slave bundle = gmem3 depth = 349696
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
  frameToLevel0(bgr, tensor, pyr + off[0], src_rows, src_cols, rows[0], cols[0], input_shift);
levels:
  for (int l = 1; l < kPyrLevels; ++l) pyramidLevel(pyr_rd + off[l - 1], pyr + off[l], rows[l - 1], cols[l - 1], rows[l], cols[l]);
}
