#include "free.h"

#include <ATen/ATen.h>
#include <Python.h>
#include <torch/extension.h>
#include <torch/script.h>

void tensor_free(torch::Tensor t) {
  auto t_data = t.data_ptr();

  free(t_data);

  return;
}
