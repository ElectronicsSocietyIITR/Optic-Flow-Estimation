// Preprocessing kernel (spec section 3; pipelined design section 7.3): one camera frame in, the
// processing frame's three products out, in one streaming pass:
//   - the DPU's input tensor: RGB, HWC, INT8 = round(v * 2^input_shift / 255) (the xmodel's input scale);
//   - gray level 0 (OpenCV's integer BGR->gray, so it equals cv::cvtColor bit for bit);
//   - gray levels 1 and 2 (Gaussian pyramid, Vitis pyrDown), shared by the LK kernel and the Horn engine.
// The resize is Vitis Vision's bilinear resize.
#pragma once

#include <ap_int.h>

constexpr int kSrcMaxRows = 1080;
constexpr int kSrcMaxCols = 1920;
constexpr int kDstMaxRows = 512;
constexpr int kDstMaxCols = 512;

// bgr: one pixel per word, B in bits 0-7, G 8-15, R 16-23. gray1/gray2: (rows + 1) / 2 per level.
void preprocess(const ap_uint<32>* bgr, ap_int<8>* tensor, ap_uint<8>* gray0, ap_uint<8>* gray1, ap_uint<8>* gray2, int src_rows,
                int src_cols, int dst_rows, int dst_cols, int input_shift);
