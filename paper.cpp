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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double getustime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double t = tv.tv_sec * 1e6 + tv.tv_usec;
    return t;
}

py::tuple optimize_sepkernels_py(py::array_t<float, py::array::c_style | py::array::forcecast> fitmols, py::array_t<float, py::array::c_style | py::array::forcecast> refmol, int gpuID, bool timingActivated) {

    // Load molecules from input numpy arrays
    auto fitmols_info = fitmols.request();
    auto refmol_info = refmol.request();

    CUDAmol cudaRefMol;
    cudaRefMol.numAtoms = refmol_info.shape[0];
    cudaRefMol.atoms = new float4[cudaRefMol.numAtoms];
    cudaMemcpy(cudaRefMol.atoms, refmol_info.ptr, cudaRefMol.numAtoms * sizeof(float4), cudaMemcpyHostToDevice);

    uint numFitMols = fitmols_info.shape[0];
    uint numFitAtoms = fitmols_info.shape[1];
    uint numTransforms = fitmols_info.shape[2];

    CUDAmol *cudaFitMols = new CUDAmol[numFitMols];
    uint *molIds = new uint[numFitMols];
    float *transforms = new float[numFitMols * numTransforms];

    for (uint i = 0; i < numFitMols; i++) {
        CUDAmol cudaFitMol;
        cudaFitMol.numAtoms = numFitAtoms;
        cudaFitMol.atoms = new float4[numFitAtoms];
        cudaMemcpy(cudaFitMol.atoms, fitmols_info.ptr + i * numFitAtoms * 4, numFitAtoms * sizeof(float4), cudaMemcpyHostToDevice);
        cudaFitMols[i] = cudaFitMol;

        for (uint j = 0; j < numTransforms; j++) {
            transforms[i * numTransforms + j] = fitmols_info.ptr[i * numFitAtoms * numTransforms + j];
        }

        molIds[i] = i;
    }

    // Move data to the GPU
    cudaSetDevice(gpuID);
    fprintf(stderr, "# Executing on GPU %d\n", gpuID);

    dCUDAMultimol devFitMM, devRefMM;
    devFitMM.numMols = numFitMols;
    devFitMM.mols = cudaFitMols;
    devRefMM.numMols = 1;
    devRefMM.mols = &cudaRefMol;

    cudaMemcpyToSymbol(dCUDAMM, &devFitMM, sizeof(dCUDAMultimol));
    cudaMemcpyToSymbol(dCUDAMMRef, &devRefMM, sizeof(dCUDAMultimol));
    cudaMemset(dDeviceOverlaps, 0, numFitMols * sizeof(float));
    cudaMemset(dDeviceTransforms, 0, numFitMols *numTransforms * sizeof(float));
// Allocate memory for results
float *dScore;
cudaMalloc(&dScore, numFitMols * sizeof(float));
float *dGrad;
cudaMalloc(&dGrad, numFitMols * numTransforms * sizeof(float));

// Set up block and grid sizes for kernel launch
int threadsPerBlock = 256;
int numBlocks = (numFitMols + threadsPerBlock - 1) / threadsPerBlock;

// Start timer
double t0 = getustime();

// Launch kernel to calculate overlaps and gradients
sepkernels<<<numBlocks, threadsPerBlock>>>(dScore, dGrad, dDeviceOverlaps, dDeviceTransforms, molIds, numFitMols, numTransforms);

// Wait for kernel to finish
cudaDeviceSynchronize();

// Stop timer
double t1 = getustime();

// Copy results back to host
float *score = new float[numFitMols];
cudaMemcpy(score, dScore, numFitMols * sizeof(float), cudaMemcpyDeviceToHost);
float *grad = new float[numFitMols * numTransforms];
cudaMemcpy(grad, dGrad, numFitMols * numTransforms * sizeof(float), cudaMemcpyDeviceToHost);

// Free device memory
cudaFree(dScore);
cudaFree(dGrad);
for (uint i = 0; i < numFitMols; i++) {
    delete[] cudaFitMols[i].atoms;
}
delete[] cudaFitMols;
delete[] cudaRefMol.atoms;
delete[] molIds;
delete[] transforms;

// Return results
py::tuple results = py::make_tuple(py::array_t<float>(numFitMols, score), py::array_t<float>(numFitMols, numTransforms, grad));

// Print timing info
if (timingActivated) {
    fprintf(stderr, "# Total execution time on GPU %d: %f ms\n", gpuID, (t1 - t0) / 1000.0);
}

return results;
}
