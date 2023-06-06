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

#define ELTS7(x) x[0],x[1],x[2],x[3],x[4],x[5],x[6]

	double getustime(void) { // {{{
        struct timeval tv;
            gettimeofday(&tv,NULL);
                double t = tv.tv_sec*1e6 + tv.tv_usec;
                    return t;
                    }

	int run_optimization(int argc, char** argv) {
        if (argc < 3) {
                    printf("paper [GPU ID] [listing file] \n");
                            printf("or\n");
                                    printf("paper [GPU ID] [reference sdf] [fit sdf] [[fit sdf] ...] \n");
                                            return 1;
                                                }
            const int gpuID = atoi(argv[1]);
            cudaSetDevice(gpuID);
            fprintf(stderr,"# Executing on GPU %d\n",gpuID);
            
            CUDAmol refmol;
            CUDAmol* fitmols;
            float3 com_ref,*com_fit;
            dCUDAMultimol hostRefMM,hostFitMM,devRefMM,devFitMM;
            uint* molids;
            uint totalMols,distinctMols;
            float* transforms;
            size_t transform_pitch;

            loadMolecules(argc-1,argv+1,
                      &fitmols,refmol,&molids,&transforms,transform_pitch,
                        hostFitMM,devFitMM,
                        hostRefMM,devRefMM,
                        com_ref,&com_fit,
                        totalMols,distinctMols);
            uint nfitmols = totalMols;

            float* hostDeviceOverlaps = (float*)malloc(nfitmols*sizeof(float));
            const int numTimers = 8;
            bool timingActivated = false;
            clock_t* hostTimings = (clock_t*)malloc(numTimers*sizeof(clock_t));
            float* hostDeviceTransforms = (float*)malloc(nfitmols*devFitMM.transform_pitch*sizeof(float));
            double optstart = getustime();
            const int itercount = optimize_sepkernels(devFitMM,devRefMM,hostDeviceOverlaps,hostTimings,numTimers,com_ref,com_fit);
            double optend = getustime();

            fprintf(stderr,"# Optimization used %d iterations of BFGS\n",itercount);
            cudaMemcpy(hostDeviceTransforms,devFitMM.transforms,nfitmols*transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);

            float* bestOverlaps = new float[distinctMols];
            float* bestTransforms = new float[distinctMols*7];

            for (uint i = 0; i < totalMols; i++) {
                        uint molid = molids[i];
                        if (hostDeviceOverlaps[i] > bestOverlaps[molid]) {
                                        bestOverlaps[molid] = hostDeviceOverlaps[i];
                                        memcpy(bestTransforms+molid*7,hostDeviceTransforms+i*transform_pitch,7*sizeof(float));
                                        }
                                    }
            if (timingActivated) {
                            printf("Size of clock_t on host side is %d\n",(int)sizeof(clock_t));
                            for (int i = 0; i < numTimers; i++) {
                            printf("Timer %d: %lld\n",i,(long long)(hostTimings[i]));
                            }
                            printf("Average clocks per operation:\n");
                            printf("   Line-search: %f\n",(double)(hostTimings[0])/hostTimings[1]);
                            printf("   Objective: %f * %f = %f\n",(double)(hostTimings[2])/hostTimings[3],(double)(hostTimings[3])/hostTimings[1],(double)(hostTimings[2])/hostTimings[1]);
                            printf("   BFGS update: %f\n",(double)(hostTimings[4])/hostTimings[5]);
                            printf("   Gradient: %f\n",(double)(hostTimings[6])/hostTimings[7]);
                    }
            bool showresults = true;
            bool benchmark   = false; 
            if (showresults) {
                    for (uint i = 0; i < nfitmols; i++) {
                        }
                    for (uint i = 0; i < distinctMols; i++) {
                        float* matrix = transformToCompensatedMatrix(bestTransforms+i*7,com_ref,com_fit[i]);
                        printTransformMatrix(matrix,stdout);
                        free(matrix);
                        }
                    }
            if (benchmark) {
                    uint bench_runs = 10;
                    double start = getustime();
                    for (uint i = 0; i < bench_runs; i++) {
                        cudaMemcpy(devRefMM.mols,hostRefMM.mols,4*hostRefMM.nmols*devRefMM.pitch*sizeof(float),cudaMemcpyHostToDevice);
                        cudaMemcpy(devRefMM.atomcounts,hostRefMM.atomcounts,1*sizeof(uint),cudaMemcpyHostToDevice);

                        cudaMemcpy(devFitMM.mols,hostFitMM.mols,4*hostFitMM.nmols*devFitMM.pitch*sizeof(float),cudaMemcpyHostToDevice);
                        cudaMemcpy(devFitMM.atomcounts,hostFitMM.atomcounts,hostFitMM.nmols*sizeof(uint),cudaMemcpyHostToDevice);
                        cudaMemcpy(devFitMM.molids,hostFitMM.molids,hostFitMM.nmols*sizeof(uint),cudaMemcpyHostToDevice);
                        cudaMemcpy(devFitMM.transforms,hostFitMM.transforms,nfitmols*transform_pitch*sizeof(float),cudaMemcpyHostToDevice);

                        const int itercount = optimize_sepkernels(devFitMM,devRefMM,hostDeviceOverlaps,hostTimings,numTimers,com_ref,com_fit);

                        cudaMemcpy(hostDeviceTransforms,devFitMM.transforms,nfitmols*transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);

                        cudaThreadSynchronize();
                        memset(bestOverlaps,0,distinctMols*sizeof(float));

                        for (uint i = 0; i < totalMols; i++) {
                                            uint molid = molids[i];
                                                    if (hostDeviceOverlaps[i] > bestOverlaps[molid]) {
                                                        bestOverlaps[molid] = hostDeviceOverlaps[i];
                                                            memcpy(bestTransforms+molid*7,hostDeviceTransforms+i*transform_pitch,7*sizeof(float));
                                            }
    }
                        }
                    double end = getustime();
                            double runtime = ((end-start)/1000)/bench_runs;
                            printf("Benchmark results over %d iterations on %d molecules (%d mol/starts): %f ms/batch optimization, %f ms/molecule, %f ms/position\n",bench_runs,distinctMols,totalMols,runtime,runtime/distinctMols,runtime/totalMols);

                            }
            delete[] bestOverlaps;
            delete[] bestTransforms;
            return 0;


        }
        

