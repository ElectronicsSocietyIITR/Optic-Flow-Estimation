#include "ttc_engine.hpp"

// Raster scan. At stream position (x, y) the 3x3 window of S = prev + curr is complete for the
// centre pixel (x - 1, y - 1); Et and the coordinates are taken for that same centre.
void ttc_engine(hls::stream<Pixel>& in, const BoxRect boxes[kMaxBoxes], const BoxRect flow_boxes[kMaxBoxes],
                const FitQ fits[kMaxBoxes], ap_uint<kMaxBoxes> fit_valid, ap_uint<16> inlier_q, ap_uint<5> n_boxes,
                ap_uint<10> width, ap_uint<10> height, ap_int<11> px, ap_int<11> py, ap_uint<13> threshold,
                ap_int<64> sums[kMaxBoxes][kTerms]) {
#pragma HLS INTERFACE axis port = in
#pragma HLS INTERFACE s_axilite port = boxes bundle = control
#pragma HLS INTERFACE s_axilite port = flow_boxes bundle = control
#pragma HLS INTERFACE s_axilite port = fits bundle = control
#pragma HLS INTERFACE s_axilite port = fit_valid bundle = control
#pragma HLS INTERFACE s_axilite port = inlier_q bundle = control
#pragma HLS INTERFACE s_axilite port = n_boxes bundle = control
#pragma HLS INTERFACE s_axilite port = width bundle = control
#pragma HLS INTERFACE s_axilite port = height bundle = control
#pragma HLS INTERFACE s_axilite port = px bundle = control
#pragma HLS INTERFACE s_axilite port = py bundle = control
#pragma HLS INTERFACE s_axilite port = threshold bundle = control
#pragma HLS INTERFACE m_axi port = sums offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = return bundle = control

  BoxRect box[kMaxBoxes], fbox[kMaxBoxes];
#pragma HLS ARRAY_PARTITION variable = box complete
#pragma HLS ARRAY_PARTITION variable = fbox complete
  // Fit predictions, adders only: row_u/v[b] = P at (x = 0, y), pred_u/v[b] = P at the current pixel.
  ap_int<64> a1[kMaxBoxes], a2[kMaxBoxes], b1[kMaxBoxes], b2[kMaxBoxes];
  ap_int<64> row_u[kMaxBoxes], row_v[kMaxBoxes], pred_u[kMaxBoxes], pred_v[kMaxBoxes];
#pragma HLS ARRAY_PARTITION variable = a1 complete
#pragma HLS ARRAY_PARTITION variable = a2 complete
#pragma HLS ARRAY_PARTITION variable = b1 complete
#pragma HLS ARRAY_PARTITION variable = b2 complete
#pragma HLS ARRAY_PARTITION variable = row_u complete
#pragma HLS ARRAY_PARTITION variable = row_v complete
#pragma HLS ARRAY_PARTITION variable = pred_u complete
#pragma HLS ARRAY_PARTITION variable = pred_v complete
  const ap_uint<32> lim = ap_uint<32>(inlier_q) * inlier_q;
  for (int b = 0; b < kMaxBoxes; ++b) {
    box[b] = boxes[b];
    fbox[b] = flow_boxes[b];
    a1[b] = fits[b].a1;
    a2[b] = fits[b].a2;
    b1[b] = fits[b].b1;
    b2[b] = fits[b].b2;
    // P(0, 0) = a0 + a1 (0 - px) + a2 (0 - py): the only multiplies, once per frame.
    row_u[b] = fits[b].a0 - fits[b].a1 * px - fits[b].a2 * py;
    row_v[b] = fits[b].b0 - fits[b].b1 * px - fits[b].b2 * py;
  }

  ap_int<64> acc[kMaxBoxes][kTerms];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
  for (int b = 0; b < kMaxBoxes; ++b)
    for (int k = 0; k < kTerms; ++k) acc[b][k] = 0;

  ap_uint<9> line_s[2][kMaxWidth];  // S for rows y-2 and y-1
#pragma HLS ARRAY_PARTITION variable = line_s complete dim = 1
  ap_int<9> line_d[kMaxWidth];      // Et = curr - prev for row y-1
  ap_uint<9> win[3][3];             // window of S: [row y-2, y-1, y][column x-2, x-1, x]
#pragma HLS ARRAY_PARTITION variable = win complete dim = 0
  ap_int<9> d_centre = 0;           // Et at (x-1, y-1)

rows:
  for (int y = 0; y < height; ++y) {
  cols:
    for (int x = 0; x < width; ++x) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 512 max = 512
      const Pixel p = in.read();

      // Flow moments need no neighbours: accumulate at the stream position itself.
      {
        if (x == 0)
          for (int b = 0; b < kMaxBoxes; ++b) {
            pred_u[b] = row_u[b];
            pred_v[b] = row_v[b];
            row_u[b] += a2[b];  // next row
            row_v[b] += b2[b];
          }
        const ap_int<11> fx = x - px, fy = y - py;
        ap_int<64> fm[kFlowTerms];
#pragma HLS ARRAY_PARTITION variable = fm complete
        fm[0] = 1;
        fm[1] = fx;
        fm[2] = fy;
        fm[3] = ap_int<64>(fx) * fx;
        fm[4] = ap_int<64>(fx) * fy;
        fm[5] = ap_int<64>(fy) * fy;
        fm[6] = p.u;
        fm[7] = p.v;
        fm[8] = ap_int<64>(fx) * p.u;
        fm[9] = ap_int<64>(fy) * p.u;
        fm[10] = ap_int<64>(fx) * p.v;
        fm[11] = ap_int<64>(fy) * p.v;
        for (int b = 0; b < kMaxBoxes; ++b) {
          // Residual to the fit's prediction in 1/64 px (floor shift, clamped to int16), the 1 px test.
          const ap_int<64> du64 = ((ap_int<64>(p.u) << kFitShift) - pred_u[b]) >> kFitShift;
          const ap_int<64> dv64 = ((ap_int<64>(p.v) << kFitShift) - pred_v[b]) >> kFitShift;
          const ap_int<17> du = du64 > 32767 ? ap_int<64>(32767) : du64 < -32767 ? ap_int<64>(-32767) : du64;
          const ap_int<17> dv = dv64 > 32767 ? ap_int<64>(32767) : dv64 < -32767 ? ap_int<64>(-32767) : dv64;
          const bool inlier = !fit_valid[b] || ap_uint<32>(du * du + dv * dv) < lim;
          pred_u[b] += a1[b];  // next pixel
          pred_v[b] += b1[b];
          const bool inside = b < n_boxes && x >= fbox[b].x1 && x < fbox[b].x2 && y >= fbox[b].y1 && y < fbox[b].y2;
          if (inside && inlier)
            for (int k = 0; k < kFlowTerms; ++k) acc[b][kHornTerms + k] += fm[k];
        }
      }
      const ap_uint<9> s_new = p.prev + p.curr;
      const ap_int<9> d_new = ap_int<9>(p.curr) - ap_int<9>(p.prev);

      const ap_uint<9> s_top = line_s[0][x], s_mid = line_s[1][x];
      line_s[0][x] = s_mid;
      line_s[1][x] = s_new;
      const ap_int<9> d_above = line_d[x];  // row y-1, column x
      line_d[x] = d_new;

      for (int r = 0; r < 3; ++r) {
        win[r][0] = win[r][1];
        win[r][1] = win[r][2];
      }
      win[0][2] = s_top;
      win[1][2] = s_mid;
      win[2][2] = s_new;

      const ap_int<9> et = d_centre;  // row y-1, column x-1 (saved last iteration)
      d_centre = d_above;
      if (y < 2 || x < 2) continue;

      const ap_int<12> ex = (ap_int<12>(win[0][2]) + 2 * win[1][2] + win[2][2]) - (ap_int<12>(win[0][0]) + 2 * win[1][0] + win[2][0]);
      const ap_int<12> ey = (ap_int<12>(win[2][0]) + 2 * win[2][1] + win[2][2]) - (ap_int<12>(win[0][0]) + 2 * win[0][1] + win[0][2]);
      const ap_uint<13> mag = (ex < 0 ? ap_int<13>(-ex) : ap_int<13>(ex)) + (ey < 0 ? ap_int<13>(-ey) : ap_int<13>(ey));
      if (mag <= threshold) continue;

      const int cx = x - 1, cy = y - 1;
      const ap_int<11> rx = cx - px, ry = cy - py;
      const ap_int<22> g = rx * ex + ry * ey;
      // Products computed once per pixel, shared by every box.
      ap_int<64> prod[kHornTerms];
#pragma HLS ARRAY_PARTITION variable = prod complete
      prod[0] = ap_int<64>(ex) * ex;
      prod[1] = ap_int<64>(ex) * ey;
      prod[2] = ap_int<64>(ex) * g;
      prod[3] = ap_int<64>(ey) * ey;
      prod[4] = ap_int<64>(ey) * g;
      prod[5] = ap_int<64>(g) * g;
      prod[6] = ap_int<64>(ex) * et;
      prod[7] = ap_int<64>(ey) * et;
      prod[8] = ap_int<64>(g) * et;
      prod[9] = ap_int<64>(et) * et;
      prod[10] = 1;
      for (int b = 0; b < kMaxBoxes; ++b) {
        const bool inside = b < n_boxes && cx >= box[b].x1 && cx < box[b].x2 && cy >= box[b].y1 && cy < box[b].y2;
        if (inside)
          for (int k = 0; k < kHornTerms; ++k) acc[b][k] += prod[k];
      }
    }
  }

  for (int b = 0; b < kMaxBoxes; ++b)
    for (int k = 0; k < kTerms; ++k) sums[b][k] = acc[b][k];
}

void ttc_engine_top(const ap_uint<8>* prev, const ap_uint<8>* curr, const ap_uint<32>* flow, const BoxRect boxes[kMaxBoxes],
                    const BoxRect flow_boxes[kMaxBoxes], const FitQ fits[kMaxBoxes], ap_uint<kMaxBoxes> fit_valid,
                    ap_uint<16> inlier_q, ap_uint<5> n_boxes, ap_uint<10> width, ap_uint<10> height, ap_int<11> px, ap_int<11> py,
                    ap_uint<13> threshold, ap_int<64> sums[kMaxBoxes][kTerms]) {
#pragma HLS INTERFACE m_axi port = prev offset = slave bundle = gmem0 depth = 262144
#pragma HLS INTERFACE m_axi port = curr offset = slave bundle = gmem1 depth = 262144
#pragma HLS INTERFACE m_axi port = flow offset = slave bundle = gmem2 depth = 262144
#pragma HLS INTERFACE m_axi port = sums offset = slave bundle = gmem3
#pragma HLS INTERFACE s_axilite port = boxes bundle = control
#pragma HLS INTERFACE s_axilite port = flow_boxes bundle = control
#pragma HLS INTERFACE s_axilite port = fits bundle = control
#pragma HLS INTERFACE s_axilite port = fit_valid bundle = control
#pragma HLS INTERFACE s_axilite port = inlier_q bundle = control
#pragma HLS INTERFACE s_axilite port = n_boxes bundle = control
#pragma HLS INTERFACE s_axilite port = width bundle = control
#pragma HLS INTERFACE s_axilite port = height bundle = control
#pragma HLS INTERFACE s_axilite port = px bundle = control
#pragma HLS INTERFACE s_axilite port = py bundle = control
#pragma HLS INTERFACE s_axilite port = threshold bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW
  hls::stream<Pixel> pixels("pixels");
#pragma HLS STREAM variable = pixels depth = 64
pack:
  for (int i = 0; i < int(width) * int(height); ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 196608 max = 196608
    const ap_uint<32> w = flow[i];
    Pixel p;
    p.prev = prev[i];
    p.curr = curr[i];
    p.u = ap_int<16>(w.range(31, 16));  // ap_fixed<16,10> bits = int16 in 1/64 px
    p.v = ap_int<16>(w.range(15, 0));
    pixels.write(p);
  }
  ttc_engine(pixels, boxes, flow_boxes, fits, fit_valid, inlier_q, n_boxes, width, height, px, py, threshold, sums);
}
