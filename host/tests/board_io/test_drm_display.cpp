// The board's DRM display, the part that runs without a display: finding the card to use.
#include <gtest/gtest.h>

#include <filesystem>

#include "fcw/board.hpp"

// /dev/dri also holds the GPU's and the fabric's (zocl) cards, so the display is the card with a
// connected connector, not always card0.
TEST(DrmDisplay, ReportsWhenNoCardHasAConnectedDisplay) {
  const auto dir = std::filesystem::temp_directory_path() / "fcw_no_dri";
  std::filesystem::create_directories(dir);
  try {
    fcw::board::findDisplayCard(dir.string());
    FAIL() << "expected an error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(dir.string()), std::string::npos) << e.what();
  }
}
