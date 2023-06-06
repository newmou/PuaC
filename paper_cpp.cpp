#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// 导入原来的函数
#include "paper.cpp"

namespace py = pybind11;

// 封装函数
py::tuple optimize_sepkernels_py(py::array_t<float, py::array::c_style | py::array::forcecast> fitmols, py::array_t<float, py::array::c_style | py::array::forcecast> refmol, int gpuID, bool timingActivated) {
    return optimize_sepkernels(fitmols.mutable_data(), refmol.mutable_data(), fitmols.shape(0), fitmols.shape(1), fitmols.shape(2), gpuID, timingActivated);
}

// 定义Python模块
PYBIND11_MODULE(your_module_name, m) {
    m.def("optimize_sepkernels_py", &optimize_sepkernels_py, "Optimize separable kernels");
}
