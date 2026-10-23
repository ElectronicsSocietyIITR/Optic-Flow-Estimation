// FrameRing: the board's gray-frame ring (spec section 5.3).
#include <gtest/gtest.h>

#include <stdexcept>

#include "fcw/frame_ring.hpp"

TEST(FrameRing, SizedForLookBackPlusFramesInFlight) {
  EXPECT_EQ(fcw::FrameRing<int>(8, 3).slots(), 12);  // scale search reads 8 back; 3 frames in flight
  EXPECT_EQ(fcw::FrameRing<int>(1, 3).slots(), 5);
}

TEST(FrameRing, LooksBackByFrameIndexEvenWhenFramesWereDropped) {
  fcw::FrameRing<int> ring(8, 3);
  for (int64_t i : {0, 1, 2, 4, 5}) ring.acquire(i) = static_cast<int>(10 * i);  // frame 3 dropped by a live camera
  ASSERT_NE(ring.at(2), nullptr);
  EXPECT_EQ(*ring.at(2), 20);
  EXPECT_EQ(ring.at(3), nullptr);
  EXPECT_EQ(*ring.at(5), 50);
}

TEST(FrameRing, RefusesToOverwriteAFrameStillNeeded) {
  fcw::FrameRing<int> ring(1, 3);  // 5 slots
  for (int64_t i = 0; i < 5; ++i) ring.acquire(i) = static_cast<int>(i);
  EXPECT_THROW(ring.acquire(5), std::runtime_error);  // slot of frame 0, which perception may still read
  ring.releaseBefore(1);
  ring.acquire(5) = 5;
  EXPECT_EQ(ring.at(0), nullptr);
  EXPECT_EQ(*ring.at(5), 5);
}
