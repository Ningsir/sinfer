#include <pybind11/pybind11.h>

#include "free.h"
#include "gather.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gather_sinfer", &gather_sinfer, "gather for sinfer", py::call_guard<py::gil_scoped_release>());
    m.def("gather_mem", &gather_mem, "gather", py::call_guard<py::gil_scoped_release>());
    m.def("gather_ssd", &gather_ssd, "gather ssd", py::call_guard<py::gil_scoped_release>());
    m.def("tensor_free", &tensor_free);
    m.def("gather_sinfer1", &gather_sinfer1, "gather for sinfer", py::call_guard<py::gil_scoped_release>());
}
