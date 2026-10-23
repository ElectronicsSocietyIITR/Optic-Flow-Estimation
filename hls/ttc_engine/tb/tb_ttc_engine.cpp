// C-simulation testbench: every golden case from tools/export_golden.py --only hls must match exactly.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/ttc_engine.hpp"

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "golden"
#endif

static std::vector<uint8_t> readBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// via_top: drive the board's top function (frames and packed 32-bit LK flow from memory) instead of the stream core.
static int runCase(const std::string& name, bool via_top) {
  const std::string dir = std::string(GOLDEN_DIR) + "/" + name + "/";
  int w, h, px, py, thr;
  std::ifstream(dir + "params.txt") >> w >> h >> px >> py >> thr;
  const std::vector<uint8_t> prev = readBytes(dir + "prev.bin"), curr = readBytes(dir + "curr.bin");
  const std::vector<uint8_t> flow_bytes = readBytes(dir + "flow.bin");
  const int16_t* flow = reinterpret_cast<const int16_t*>(flow_bytes.data());  // u, v interleaved
  if (static_cast<int>(prev.size()) != w * h || static_cast<int>(curr.size()) != w * h) {
    std::printf("%s: frame size mismatch\n", name.c_str());
    return 1;
  }
  BoxRect boxes[kMaxBoxes] = {};
  int n = 0;
  std::ifstream bf(dir + "boxes.txt");
  for (int x1, y1, x2, y2; bf >> x1 >> y1 >> x2 >> y2;) boxes[n++] = {x1, y1, x2, y2};
  BoxRect flow_boxes[kMaxBoxes] = {};
  int nf = 0;
  std::ifstream ff(dir + "flow_boxes.txt");
  for (int x1, y1, x2, y2; ff >> x1 >> y1 >> x2 >> y2;) flow_boxes[nf++] = {x1, y1, x2, y2};

  FitQ fits[kMaxBoxes] = {};
  ap_uint<kMaxBoxes> fit_valid = 0;
  int inlier_q = 64;
  std::ifstream qf(dir + "fits.txt");
  qf >> inlier_q;
  for (int b = 0; b < nf; ++b) {
    long long valid, c[6];
    if (!(qf >> valid >> c[0] >> c[1] >> c[2] >> c[3] >> c[4] >> c[5])) break;
    fits[b] = {c[0], c[1], c[2], c[3], c[4], c[5]};
    fit_valid[b] = valid != 0;
  }

  static ap_int<64> sums[kMaxBoxes][kTerms];
  if (via_top) {
    // The LK kernel's flow word: u in the upper 16 bits, v in the lower 16 (ap_fixed<16,10>, 1/64 px).
    std::vector<ap_uint<8>> p8(prev.begin(), prev.end()), c8(curr.begin(), curr.end());
    std::vector<ap_uint<32>> packed(w * h);
    for (int i = 0; i < w * h; ++i) packed[i] = (ap_uint<32>(ap_uint<16>(flow[2 * i])) << 16) | ap_uint<16>(flow[2 * i + 1]);
    ttc_engine_top(p8.data(), c8.data(), packed.data(), boxes, flow_boxes, fits, fit_valid, inlier_q, n, w, h, px, py, thr, sums);
  } else {
    hls::stream<Pixel> in;
    for (int i = 0; i < w * h; ++i) in.write({prev[i], curr[i], flow[2 * i], flow[2 * i + 1]});
    ttc_engine(in, boxes, flow_boxes, fits, fit_valid, inlier_q, n, w, h, px, py, thr, sums);
  }

  std::ifstream ef(dir + "expected.txt");
  int failures = 0;
  for (int b = 0; b < n; ++b) {
    for (int k = 0; k < kTerms; ++k) {
      long long expected;
      ef >> expected;
      const long long got = sums[b][k].to_int64();
      if (got != expected && failures++ < 5)
        std::printf("%s box %d term %d: got %lld expected %lld\n", name.c_str(), b, k, got, expected);
    }
  }
  std::printf("%-18s %d boxes %s %s\n", name.c_str(), n, via_top ? "top   " : "stream", failures ? "FAIL" : "exact");
  return failures ? 1 : 0;
}

int main() {
  int failed = 0;
  for (const char* c : {"random_overlap", "random_threshold", "approach", "checker", "flat", "masked", "level1"}) failed += runCase(c, false);
  for (const char* c : {"approach", "masked"}) failed += runCase(c, true);
  std::printf(failed ? "C-SIM FAILED (%d cases)\n" : "C-SIM PASSED\n", failed);
  return failed ? 1 : 0;
}
