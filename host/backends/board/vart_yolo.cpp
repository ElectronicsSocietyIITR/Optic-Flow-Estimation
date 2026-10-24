// YOLOv9t on the DPU through VART (see fcw/board.hpp). Board only: needs the Vitis AI runtime.
// Not yet run on a board (plan 2, Task 10).
#include <vart/runner_ext.hpp>
#include <xir/graph/graph.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "fcw/board.hpp"
#include "fcw/yolo_decode.hpp"

namespace fcw::board {

struct VartYolo::Impl {
  std::unique_ptr<xir::Graph> graph;
  std::unique_ptr<vart::RunnerExt> runner;
  std::vector<vart::TensorBuffer*> inputs, outputs;
  int input_fix = 0;
};

VartYolo::VartYolo(const std::string& xmodel) : impl_(std::make_unique<Impl>()) {
  impl_->graph = xir::Graph::deserialize(xmodel);
  const xir::Subgraph* dpu = nullptr;
  for (const xir::Subgraph* s : impl_->graph->get_root_subgraph()->children_topological_sort())
    if (s->has_attr("device") && s->get_attr<std::string>("device") == "DPU") dpu = s;  // compiled as one DPU subgraph
  if (!dpu) throw std::runtime_error("VART: no DPU subgraph in " + xmodel);
  auto attrs = xir::Attrs::create();
  impl_->runner = vart::RunnerExt::create_runner(dpu, attrs.get());
  impl_->inputs = impl_->runner->get_inputs();
  impl_->outputs = impl_->runner->get_outputs();
  impl_->input_fix = impl_->inputs[0]->get_tensor()->get_attr<int>("fix_point");
}

VartYolo::~VartYolo() = default;

int VartYolo::inputShift() const { return impl_->input_fix; }

std::vector<Detection> VartYolo::operator()(const int8_t* tensor_hwc, cv::Size input) {
  Impl& d = *impl_;
  vart::TensorBuffer* in = d.inputs[0];
  const auto [addr, size] = in->data({0, 0, 0, 0});
  std::memcpy(reinterpret_cast<void*>(addr), tensor_hwc, std::min<size_t>(size, static_cast<size_t>(input.area()) * 3));
  in->sync_for_write(0, size);
  const auto job = d.runner->execute_async(d.inputs, d.outputs);
  d.runner->wait(static_cast<int>(job.first), -1);
  // Raw head maps (N, H, W, 144) INT8 -> float, finest stride first, as decodeYolo takes them.
  std::vector<cv::Mat> heads;
  for (vart::TensorBuffer* out : d.outputs) {
    const auto shape = out->get_tensor()->get_shape();
    const int h = shape[1], w = shape[2], c = shape[3];
    const float scale = std::ldexp(1.0f, -out->get_tensor()->get_attr<int>("fix_point"));
    const auto [oaddr, osize] = out->data({0, 0, 0, 0});
    out->sync_for_read(0, osize);
    const int8_t* q = reinterpret_cast<const int8_t*>(oaddr);
    cv::Mat m(h, w, CV_32FC(c));
    float* f = m.ptr<float>();
    for (size_t i = 0; i < static_cast<size_t>(h) * w * c; ++i) f[i] = q[i] * scale;
    heads.push_back(m);
  }
  std::sort(heads.begin(), heads.end(), [](const cv::Mat& a, const cv::Mat& b) { return a.cols > b.cols; });
  return decodeYolo(heads, input);
}

}  // namespace fcw::board
