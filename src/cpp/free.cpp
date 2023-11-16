#include <torch/extension.h>
#include <Python.h>
#include <torch/script.h>
#include <ATen/ATen.h>

#include "free.h"

void tensor_free(torch::Tensor t)
{
    auto t_data = t.data_ptr();

    free(t_data);

    return;
}
