#include <pybind11/pybind11.h>

#include "free.h"
#include "gather.h"
#include "spdlog/cfg/env.h"
#include "store.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_spdlog",
        &spdlog::cfg::load_env_levels,
        "init spdlog with env variable");
  m.def("gather_cache_ssd_dma",
        &gather_cache_ssd_dma,
        "在缓存和ssd(利用DMA)中读取数据",
        py::call_guard<py::gil_scoped_release>());
  m.def("gather_range",
        &gather_range,
        "读取顶点ID在[start, end)范围内的数据",
        py::call_guard<py::gil_scoped_release>());
  m.def("gather_ssd",
        &gather_ssd,
        "从磁盘读取数据",
        py::call_guard<py::gil_scoped_release>());
  m.def("tensor_free", &tensor_free);
  m.def("gather_cache_ssd",
        &gather_cache_ssd,
        "在缓存和内存中的数据分开执行gather",
        py::call_guard<py::gil_scoped_release>());

  py::class_<FeatureStore, std::shared_ptr<FeatureStore>>(m, "FeatureStore")
      .def(py::init<>([](const std::string file_path,
                         std::vector<int64_t> offsets,
                         int64_t num,
                         int64_t dim,
                         bool prefetch,
                         py::object dtype,
                         int num_writer_workers,
                         bool writer_seq) {
             return std::make_shared<FeatureStore>(
                 file_path,
                 offsets,
                 num,
                 dim,
                 prefetch,
                 torch::python::detail::py_object_to_dtype(dtype),
                 num_writer_workers,
                 writer_seq);
           }),
           py::arg("file_path"),
           py::arg("offsets"),
           py::arg("num"),
           py::arg("dim"),
           py::arg("prefetch") = true,
           py::arg("dtype") = torch::kFloat32,
           py::arg("num_writer_workers") = 2,
           py::arg("writer_seq") = true)
      .def("update_cache",
           &FeatureStore::update_cache,
           py::call_guard<py::gil_scoped_release>())
      .def("gather",
           &FeatureStore::gather,
           py::call_guard<py::gil_scoped_release>())
       .def("gather_all",
           &FeatureStore::gather_all,
           py::call_guard<py::gil_scoped_release>())
      .def("write_data",
           &FeatureStore::write_data,
           py::call_guard<py::gil_scoped_release>())
      .def("flush",
           &FeatureStore::flush,
           py::call_guard<py::gil_scoped_release>());
}
