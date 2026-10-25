// Horn TTC engine (spec section 4.8; pipelined design section 7): one streaming pass over a frame
// pair, per-box Horn sums and flow moments for up to 16 boxes. Bit-exact with fcw::hornSums and
// fcw::flowMomentsFixed / predictedMaskFixed (host/src) and their Python twins.
#pragma once

#include <ap_int.h>
#include <hls_stream.h>

constexpr int kMaxWidth = 512;
constexpr int kMaxHeight = 512;
#ifndef TTC_LANES
#define TTC_LANES 4  // boxes per start: 4 lanes route at 15.2k LUTs (16: ~35k); the host batches (planEnginePasses)
#endif
constexpr int kMaxBoxes = TTC_LANES;
// 10 Horn sums (Ex^2, ExEy, ExG, Ey^2, EyG, G^2, ExEt, EyEt, GEt, Et^2) + Horn pixel count,
// then 12 flow moments (n, x, y, xx, xy, yy, u, v, xu, yu, xv, yv; u, v in 1/64 px).
constexpr int kHornTerms = 11;
constexpr int kFlowTerms = 12;
constexpr int kTerms = kHornTerms + kFlowTerms;
constexpr int kFitShift = 16;  // fraction bits of the affine fit (divergence.py FIT_SHIFT)

struct Pixel {
  ap_uint<8> prev;
  ap_uint<8> curr;
  ap_int<16> u;  // flow in 1/64 px
  ap_int<16> v;
};

// Integer inner rectangle [x1, x2) x [y1, y2), from fcw::innerBox / flowRect on the host.
struct BoxRect {
  ap_uint<10> x1, y1, x2, y2;
};

// A track's affine flow fit from the previous frame, predicting this frame's object pixels:
// P = a0 + a1 x + a2 y in 1/64 px scaled by 2^16, x, y from the principal point.
struct FitQ {
  ap_int<64> a0, a1, a2, b0, b1, b2;
};

// Stream core. fit_valid bit b: flow box b sums only pixels within inlier_q (1/64 px) of its fit;
// otherwise every pixel of the box (the host then refits it in software, RANSAC fallback).
void ttc_engine(hls::stream<Pixel>& in, const BoxRect boxes[kMaxBoxes], const BoxRect flow_boxes[kMaxBoxes],
                const FitQ fits[kMaxBoxes], ap_uint<kMaxBoxes> fit_valid, ap_uint<16> inlier_q, ap_uint<5> n_boxes,
                ap_uint<10> width, ap_uint<10> height, ap_int<11> px, ap_int<11> py, ap_uint<13> threshold,
                ap_int<64> sums[kMaxBoxes][kTerms]);

// Board top: reads gray t-1, gray t and the LK kernel's flow (32-bit words: u in the upper 16 bits,
// v in the lower 16, ap_fixed<16,10> = 1/64 px) from DDR, packs them into the stream and runs the core.
// Tables as plain words (no struct layout for the host to guess): boxes, flow_boxes: x1 y1 x2 y2 per box;
// fits: a0 a1 a2 b0 b1 b2 per box.
void ttc_engine_top(const ap_uint<8>* prev, const ap_uint<8>* curr, const ap_uint<32>* flow, const ap_uint<32>* boxes,
                    const ap_uint<32>* flow_boxes, const ap_int<64>* fits, ap_uint<kMaxBoxes> fit_valid, ap_uint<16> inlier_q,
                    ap_uint<5> n_boxes, ap_uint<10> width, ap_uint<10> height, ap_int<11> px, ap_int<11> py, ap_uint<13> threshold,
                    ap_int<64> sums[kMaxBoxes * kTerms]);
