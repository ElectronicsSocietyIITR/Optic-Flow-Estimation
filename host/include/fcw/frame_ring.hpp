// Ring of per-frame buffers indexed by frame number (spec section 5.3): the board's gray-frame
// and pyramid ring, written by preprocessing ahead of perception and read back up to look_back
// frames. Slots = look_back + in_flight + 1, so the writer, up to in_flight frames ahead, never
// lands on a frame perception may still read; acquire() refuses if it would.
// Indices may skip (a live camera drops frames): at() answers by frame index, not by position.
#pragma once

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fcw {

template <class Slot>
class FrameRing {
 public:
  FrameRing(int look_back, int in_flight) : tags_(look_back + in_flight + 1, -1), data_(tags_.size()) {}

  int slots() const { return static_cast<int>(tags_.size()); }

  // The slot for frame_index, to be filled by the caller. Throws if it still holds a needed frame.
  Slot& acquire(int64_t frame_index) {
    std::lock_guard<std::mutex> lock(m_);
    const size_t s = static_cast<size_t>(frame_index % slots());
    if (tags_[s] >= oldest_needed_ && tags_[s] != frame_index)
      throw std::runtime_error("FrameRing: frame " + std::to_string(tags_[s]) + " is still needed (writer too far ahead)");
    tags_[s] = frame_index;
    return data_[s];
  }

  // Frame frame_index, or nullptr when it was never written (dropped) or already overwritten.
  const Slot* at(int64_t frame_index) const {
    std::lock_guard<std::mutex> lock(m_);
    const size_t s = static_cast<size_t>(frame_index % slots());
    return tags_[s] == frame_index ? &data_[s] : nullptr;
  }

  // Frames before oldest_needed may be overwritten (perception has moved past them).
  void releaseBefore(int64_t oldest_needed) {
    std::lock_guard<std::mutex> lock(m_);
    oldest_needed_ = oldest_needed;
  }

 private:
  mutable std::mutex m_;
  std::vector<int64_t> tags_;
  std::vector<Slot> data_;
  int64_t oldest_needed_ = 0;
};

}  // namespace fcw
