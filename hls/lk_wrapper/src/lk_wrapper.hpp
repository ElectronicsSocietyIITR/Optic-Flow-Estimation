// Dense pyramidal LK for a whole frame pair in ONE kernel start (issue 6): AMD's densePyrOpticalFlow
// pass looped over 5 levels x 5 iterations on-chip, flow ping-ponged between two halves of one DDR buffer,
// instead of 25 starts from the host. Same calls, same order as AMD's L1 testbench (and hls/lk_ref),
// so the flow is bit-identical.
#pragma once

#include <ap_int.h>

#include "../../common/pyramid_layout.hpp"

#ifndef LK_MAX
#define LK_MAX 512  // largest level the kernel processes (both dimensions); -DLK_MAX=256 for the half-resolution design
#endif
constexpr int kLkMaxRows = LK_MAX;
constexpr int kLkMaxCols = LK_MAX;

// pyr_prev, pyr_curr: pyramid slots (hls/common/pyramid_layout.hpp), 4 pixels per word.
// flow_rd, flow_wr: the SAME buffer of 2 x rows x cols words (u in the upper 16 bits, v in the lower 16,
// ap_fixed<16,10>), read and written through two ports; passes ping-pong between its halves at offsets 0
// and rows x cols. The host zeroes it once. The result is in the first half (25 passes, an odd number).
// (One pass call site: two would make HLS build the whole LK core twice.)
void lk_wrapper(const ap_uint<32>* pyr_prev, const ap_uint<32>* pyr_curr, const ap_uint<32>* flow_rd, ap_uint<32>* flow_wr, int rows,
                int cols);
