// One frame's gray pyramid in DDR (a gray-ring slot): levels 0..kPyrLevels-1, each (rows+1)/2 x (cols+1)/2
// of the one above, every level starting on a 64-byte boundary. Written by hls/preprocess, read by
// hls/lk_wrapper (all levels) and hls/ttc_engine (levels 0-2), and by the host (FrameRing slots).
#pragma once

constexpr int kPyrLevels = 5;

inline int pyrRows(int rows, int level) {
  for (int l = 0; l < level; ++l) rows = (rows + 1) / 2;
  return rows;
}
inline int pyrCols(int cols, int level) { return pyrRows(cols, level); }
inline int align64(int n) { return (n + 63) / 64 * 64; }
// Byte offset of a level in the slot.
inline int pyrOffset(int rows, int cols, int level) {
  int off = 0;
  for (int l = 0; l < level; ++l) off += align64(pyrRows(rows, l) * pyrCols(cols, l));
  return off;
}
inline int pyrBytes(int rows, int cols) { return pyrOffset(rows, cols, kPyrLevels); }
