#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "inputModule.h"
#include "cudaVolumeTypes.h"
#include "hostAnalyticVolume.h"
#include "deviceAnalyticVolume.h"
#include "deviceOverlay.h"
#include "transformTools.h"
#include <sys/time.h>
#include <pybind11.h>

// 使用适当的参数类型和返回类型
pybind11::tuple wrapped_run_optimization(pybind11::args args) {
	    int argc = args.size();
	        std::vector<char*> argv;
		    for (const auto& arg : args) {
			            argv.push_back(const_cast<char*>(pybind11::str(arg).cast<std::string>().c_str()));
				        }
		        int result = run_optimization(argc, argv.data());
			    return pybind11::make_tuple(result);  // 根据需要返回值
}

PYBIND11_MODULE(cuda_paper, m) {
	m.doc() = "Pybind11 CUDA Paper module";
        m.def("run_optimization", &wrapped_run_optimization, "Run the optimization on CUDA");
}

