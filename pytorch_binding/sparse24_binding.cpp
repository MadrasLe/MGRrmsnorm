#include <torch/extension.h>

#include "../src/sparse24_fp16_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "linear",
        &sparse24_fp16_linear_cuda,
        "Standalone FP16 2:4 linear using NVIDIA mma.sp Sparse Tensor Cores",
        py::arg("input"),
        py::arg("values"),
        py::arg("metadata"),
        py::arg("bias") = c10::nullopt,
        py::arg("output") = c10::nullopt
    );
}
