// Preprocessing kernel (spec section 3; pipelined design section 7.3): one camera frame in, the
// processing frame's products out, in one streaming pass:
//   - the DPU's input tensor: RGB, HWC, INT8 = round(v * 2^input_shift / 255) (the xmodel's input scale);
//   - the gray pyramid slot (hls/common/pyramid_layout.hpp): level 0 by OpenCV's integer BGR->gray
//     (= bgr_to_gray in the pipelines), levels 1-4 by one Vitis pyrDown, level by level through DDR (= cv::pyrDown, replicated borders).
// The resize is Vitis Vision's bilinear resize.
#pragma once

#include <ap_int.h>

#include "../../common/pyramid_layout.hpp"

constexpr int kSrcMaxRows = 1080;
constexpr int kSrcMaxCols = 1920;
constexpr int kDstMaxRows = 512;
constexpr int kDstMaxCols = 512;

// bgr: one pixel per word, B in bits 0-7, G 8-15, R 16-23. pyr: pyrBytes(dst_rows, dst_cols) bytes, written;
// pyr_rd: the same buffer, read back for levels 1-4 (a second port, so reading and writing can overlap).
void preprocess(const ap_uint<32>* bgr, ap_int<8>* tensor, ap_uint<8>* pyr, const ap_uint<8>* pyr_rd, int src_rows, int src_cols,
                int dst_rows, int dst_cols, int input_shift);
