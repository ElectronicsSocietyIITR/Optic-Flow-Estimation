// ThreadPool: the object stage's worker pool.
#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

#include "fcw/thread_pool.hpp"

TEST(ThreadPool, RunsEveryIndexOnce) {
  fcw::ThreadPool pool(3);
  std::vector<std::atomic<int>> hits(100);
  pool.parallelFor(100, [&](int i) { ++hits[i]; });
  for (int i = 0; i < 100; ++i) EXPECT_EQ(hits[i].load(), 1) << i;
}

// A throw must reach the caller only after every worker has left fn (fn and its captures die with the
// caller's frame), and the pool must stay usable.
TEST(ThreadPool, PropagatesAnExceptionAfterAllWorkersFinish) {
  fcw::ThreadPool pool(3);
  std::atomic<int> running{0}, after_throw{0};
  std::atomic<bool> thrown{false};
  auto fn = [&](int i) {
    ++running;
    if (i == 0) {
      thrown = true;
      --running;
      throw std::runtime_error("object 0 failed");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // workers still busy when index 0 throws
    if (thrown) ++after_throw;
    --running;
  };
  EXPECT_THROW(pool.parallelFor(8, fn), std::runtime_error);
  EXPECT_EQ(running.load(), 0) << "a worker was still inside fn when parallelFor returned";
  std::atomic<int> count{0};
  pool.parallelFor(16, [&](int) { ++count; });
  EXPECT_EQ(count.load(), 16);
}

TEST(ThreadPool, PropagatesAnExceptionThrownOnAWorker) {
  fcw::ThreadPool pool(2);
  std::atomic<int> calls{0};
  auto fn = [&](int i) {
    ++calls;
    if (i == 5) throw std::runtime_error("object 5 failed");
  };
  EXPECT_THROW(pool.parallelFor(8, fn), std::runtime_error);
  std::atomic<int> count{0};
  pool.parallelFor(4, [&](int) { ++count; });
  EXPECT_EQ(count.load(), 4);
}
