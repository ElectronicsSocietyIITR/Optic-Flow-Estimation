// C-simulation: lk_wrapper (one kernel start) must give exactly the flow of the host-driven loop
// (hls/lk_ref, AMD's testbench sequence) on the same frame pair.
//   tb_lk_wrapper prev.png curr.png reference_flow.bin
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <fstream>
#include <vector>

#include "../src/lk_wrapper.hpp"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: tb_lk_wrapper prev.png curr.png reference_flow.bin\n");
    return 2;
  }
  const cv::Mat im[2] = {cv::imread(argv[1], cv::IMREAD_GRAYSCALE), cv::imread(argv[2], cv::IMREAD_GRAYSCALE)};
  const int rows = im[0].rows, cols = im[0].cols;
  // Pyramid slots as hls/preprocess writes them (Vitis pyrDown = cv::pyrDown with replicated borders).
  std::vector<ap_uint<32>> slot[2];
  for (int k = 0; k < 2; ++k) {
    std::vector<uint8_t> bytes(pyrBytes(rows, cols), 0);
    cv::Mat level = im[k];
    for (int l = 0; l < kPyrLevels; ++l) {
      if (l) cv::pyrDown(level, level, cv::Size(), cv::BORDER_REPLICATE);
      std::copy(level.data, level.data + level.total(), bytes.begin() + pyrOffset(rows, cols, l));
    }
    slot[k].resize(bytes.size() / 4);
    for (size_t i = 0; i < slot[k].size(); ++i)
      slot[k][i] = ap_uint<32>(bytes[4 * i]) | (ap_uint<32>(bytes[4 * i + 1]) << 8) | (ap_uint<32>(bytes[4 * i + 2]) << 16) |
                   (ap_uint<32>(bytes[4 * i + 3]) << 24);
  }
  std::vector<ap_uint<32>> a(rows * cols, 0), b(rows * cols, 0);
  lk_wrapper(slot[0].data(), slot[1].data(), a.data(), b.data(), rows, cols);

  std::vector<float> ref(rows * cols * 2);
  std::ifstream(argv[3], std::ios::binary).read(reinterpret_cast<char*>(ref.data()), ref.size() * sizeof(float));
  int diff = 0;
  for (int i = 0; i < rows * cols; ++i) {
    const short u = static_cast<short>(a[i] >> 16), v = static_cast<short>(a[i] & 0xFFFF);
    if (u / 64.0f != ref[2 * i] || v / 64.0f != ref[2 * i + 1]) ++diff;
  }
  std::printf("lk_wrapper %dx%d: %d of %d flow vectors differ from the host-driven loop -> %s\n", cols, rows, diff, rows * cols,
              diff ? "C-SIM FAILED" : "C-SIM PASSED");
  return diff ? 1 : 0;
}
