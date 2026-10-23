// Fixed set of worker threads for data-parallel loops (the per-object stage, spec section 5.2).
// parallelFor(n, fn) runs fn(0..n-1) on the workers and the calling thread and returns when all
// are done. With 0 workers everything runs on the calling thread, in order.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace fcw {

class ThreadPool {
 public:
  explicit ThreadPool(int workers) {
    for (int i = 0; i < workers; ++i) threads_.emplace_back([this] { loop(); });
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
  }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void parallelFor(int n, const std::function<void(int)>& fn) {
    if (threads_.empty() || n <= 1) {
      for (int i = 0; i < n; ++i) fn(i);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(m_);
      fn_ = &fn;
      n_ = n;
      next_.store(0);
      busy_ = static_cast<int>(threads_.size());
      ++generation_;
    }
    cv_.notify_all();
    work();  // the caller helps
    std::unique_lock<std::mutex> lock(m_);
    done_.wait(lock, [this] { return busy_ == 0; });
    fn_ = nullptr;
  }

 private:
  void work() {
    for (int i = next_.fetch_add(1); i < n_; i = next_.fetch_add(1)) (*fn_)(i);
  }
  void loop() {
    long seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
      }
      work();
      {
        std::lock_guard<std::mutex> lock(m_);
        if (--busy_ == 0) done_.notify_one();
      }
    }
  }

  std::vector<std::thread> threads_;
  std::mutex m_;
  std::condition_variable cv_, done_;
  const std::function<void(int)>* fn_ = nullptr;
  std::atomic<int> next_{0};
  int n_ = 0;
  int busy_ = 0;
  long generation_ = 0;
  bool stop_ = false;
};

}  // namespace fcw
