from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':

    setup(


        ext_modules=[
            CUDAExtension(
                name="cpp_CUDA_code.pointnet_cuda",
                sources=[
                    "paper_api.cpp",
                    "paper.cu",
                ]   
            ),
        ],
    )