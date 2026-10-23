// Dense pyramidal LK for a whole frame pair in ONE kernel start (issue 6): AMD's densePyrOpticalFlow
// pass looped over 5 levels x 5 iterations on-chip, flow ping-ponged between two DDR buffers,
// instead of 25 starts from the host. Same calls, same order as AMD's L1 testbench (and hls/lk_ref),
// so the flow is bit-identical.
#pragma once

#include <ap_int.h>

#include "../../common/pyramid_layout.hpp"

constexpr int kLkMaxRows = 512;
constexpr int kLkMaxCols = 512;

// pyr_prev, pyr_curr: pyramid slots (hls/common/pyramid_layout.hpp), 4 pixels per word.
// flow_a, flow_b: rows x cols words each (u in the upper 16 bits, v in the lower 16, ap_fixed<16,10>);
// the host zeroes them once. The result is in flow_a (25 passes, an odd number).
void lk_wrapper(const ap_uint<32>* pyr_prev, const ap_uint<32>* pyr_curr, ap_uint<32>* flow_a, ap_uint<32>* flow_b, int rows, int cols);
