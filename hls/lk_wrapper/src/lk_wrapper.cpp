#include "lk_wrapper.hpp"

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "video/xf_pyr_dense_optical_flow_wrapper.hpp"

namespace {

constexpr int kLevels = kPyrLevels;  // 5
constexpr int kIterations = 5;
constexpr int kLines = 50;           // NUM_LINES_FINDIT, as AMD's example
constexpr int kWin = 11;
constexpr int kFlowWidth = 16, kFlowInt = 10;
using Img = xf::cv::Mat<XF_8UC1, kLkMaxRows, kLkMaxCols, XF_NPPC1, 2>;
using Flow = xf::cv::Mat<XF_32UC1, kLkMaxRows, kLkMaxCols, XF_NPPC1, 2>;

// One LK pass at one level (AMD's pyr_dense_optical_flow_accel body).
void pass(const ap_uint<32>* cur, const ap_uint<32>* nxt, const ap_uint<32>* flow_in, ap_uint<32>* flow_out, int level, int scale_up,
          float scale_in, int init, int rows, int cols, int in_rows, int in_cols) {
#pragma HLS INLINE off
  Img cur_mat(rows, cols), nxt_mat(rows, cols);
  Flow fin(in_rows, in_cols), fout(rows, cols);
#pragma HLS DATAFLOW
  xf::cv::Array2xfMat<32, XF_8UC1, kLkMaxRows, kLkMaxCols, XF_NPPC1, 2>(const_cast<ap_uint<32>*>(cur), cur_mat);
  xf::cv::Array2xfMat<32, XF_8UC1, kLkMaxRows, kLkMaxCols, XF_NPPC1, 2>(const_cast<ap_uint<32>*>(nxt), nxt_mat);
  xf::cv::Array2xfMat<32, XF_32UC1, kLkMaxRows, kLkMaxCols, XF_NPPC1, 2>(const_cast<ap_uint<32>*>(flow_in), fin);
  xf::cv::densePyrOpticalFlow<kLevels, kLines, kWin, kFlowWidth, kFlowInt, XF_8UC1, kLkMaxRows, kLkMaxCols, XF_NPPC1, false, 2, 2, 2, 2>(
      cur_mat, nxt_mat, fin, fout, level, scale_up, scale_in, init);
  xf::cv::xfMat2Array<32, XF_32UC1, kLkMaxRows, kLkMaxCols, XF_NPPC1, 2>(fout, flow_out);
}

}  // namespace

void lk_wrapper(const ap_uint<32>* pyr_prev, const ap_uint<32>* pyr_curr, ap_uint<32>* flow_a, ap_uint<32>* flow_b, int rows, int cols) {
#pragma HLS INTERFACE m_axi port = pyr_prev offset = slave bundle = gmem0 depth = 87424
#pragma HLS INTERFACE m_axi port = pyr_curr offset = slave bundle = gmem1 depth = 87424
#pragma HLS INTERFACE m_axi port = flow_a offset = slave bundle = gmem2 depth = 262144
#pragma HLS INTERFACE m_axi port = flow_b offset = slave bundle = gmem3 depth = 262144
#pragma HLS INTERFACE s_axilite port = rows
#pragma HLS INTERFACE s_axilite port = cols
#pragma HLS INTERFACE s_axilite port = return
  bool into_a = true;  // the first pass writes flow_a; 25 passes end in flow_a
levels:
  for (int l = kLevels - 1; l >= 0; --l) {
    const int r = pyrRows(rows, l), c = pyrCols(cols, l), off = pyrOffset(rows, cols, l) / 4;
  iterations:
    for (int it = 0; it < kIterations; ++it) {
      const bool scale_up = it == 0 && l != kLevels - 1;
      const int in_r = scale_up ? pyrRows(rows, l + 1) : r, in_c = scale_up ? pyrCols(cols, l + 1) : c;
      const float scale_in = (in_r - 1) * 1.0f / (r - 1);
      const int init = (it == 0 && l == kLevels - 1) ? 1 : 0;
      if (into_a)
        pass(pyr_prev + off, pyr_curr + off, flow_b, flow_a, l, scale_up, scale_in, init, r, c, in_r, in_c);
      else
        pass(pyr_prev + off, pyr_curr + off, flow_a, flow_b, l, scale_up, scale_in, init, r, c, in_r, in_c);
      into_a = !into_a;
    }
  }
}
