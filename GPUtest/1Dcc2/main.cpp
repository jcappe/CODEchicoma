

//-----------------------------------------------------------

#include <math.h>

#include <iostream>
#include <malloc.h>

#include <chrono>

#include <cuda_runtime.h>
#include <cufftXt.h>
#include "cufft_utils.h"


//-----------------------------------------------------------
int main() 
{

    srand(std::chrono::system_clock::now().time_since_epoch().count());

    std::chrono::time_point<std::chrono::system_clock> t0s, t0e, t1s, t1e, t2s, t2e, t3s, t3e, t4s, t4e, t5s, t5e;
    std::chrono::duration<double> t0se, t1se, t2se, t3se, t4se, t5se;

    t0s = std::chrono::system_clock::now();


    t1s = std::chrono::system_clock::now();

    int long long i;
    double n2pow = 6;
    int long long n = pow(2.0,n2pow);

    double xstart = 0.0;
    double xend = 1.0;
    double f = 2.0;
    double pi = 3.14159;

    float* data = (float*)calloc(sizeof(float)*2,n);
    for (i = 0; i < n; i++) 
    {
        data[2*i+0] = cos(2*pi*f*(xend-xstart)/((float)n-1)*(float)i);
        data[2*i+1] = 0.0f;
    }

    t1e = std::chrono::system_clock::now();

    //-----------------------------------------------------------
    // PRINT INPUT
    printf("\n");
    printf("Input array (complex valued):\n");
    for (i=0;i<n;i++)
    {
        printf("%f, %f\n", data[2*i], data[2*i+1]);
    }
    printf("\n");


    t2s = std::chrono::system_clock::now();
    //-----------------------------------------------------------
    // CREATE PLAN
    cufftHandle plan;
    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlan1d(&plan, n, CUFFT_C2C, 1));

    // CREATE STREAM
    cudaStream_t stream = NULL;
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan, stream));

    //-----------------------------------------------------------
    cufftComplex *d_data = nullptr;

    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    t2e = std::chrono::system_clock::now();

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //  CALCULATIONS
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%

    t3s = std::chrono::system_clock::now();
    //-----------------------------------------------------------
    // Allocate and Copy Memory
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(float) * 2 * n));
    CUDA_RT_CALL(cudaMemcpyAsync(d_data, data, sizeof(float) * 2 * n, cudaMemcpyHostToDevice, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    t3e = std::chrono::system_clock::now();

    t4s = std::chrono::system_clock::now();
    //-----------------------------------------------------------
    // Execute Compute Functions
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CUDA_RT_CALL(cudaMemcpyAsync(data, d_data, sizeof(float) * 2 * n, cudaMemcpyDeviceToHost, stream));  // put copy here if want to see transform data on GPU
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    t4e = std::chrono::system_clock::now();
    
    //t5s = std::chrono::system_clock::now();
    //CUDA_RT_CALL(cudaMemcpyAsync(data, d_data, sizeof(float) * 2 * n, cudaMemcpyDeviceToHost, stream));     // put copy here if want to see inverse transform data (should be back to original intput multiplied by number of elements)

    //-----------------------------------------------------------
    // Synchronize CPU and GPU (wait for GPU to finish all work assigned)
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    //t5e = std::chrono::system_clock::now();

    //-----------------------------------------------------------
    // FREE
    CUDA_RT_CALL(cudaFree(d_data))
    CUFFT_CALL(cufftDestroy(plan));
    CUDA_RT_CALL(cudaStreamDestroy(stream));
    CUDA_RT_CALL(cudaDeviceReset());


    //-----------------------------------------------------------
    // PRINT OUTPUT
    printf("Output array (complex valued):\n");
    for (i=0;i<n;i++)
    {
        printf("%f, %f\n", data[2*i]/(double)n, data[2*i+1]/(double)n);
    }
    printf("\n");


    t0e = std::chrono::system_clock::now();


    //-----------------------------------------------------------
    // PRINT TIMING RESULTS
    
    printf("\n");
    t0se = t0e-t0s;
    printf("DURATION 0 (seconds): EVERYTHING = %e\n",t0se);
    t1se = t1e-t1s;
    printf("DURATION 1 (seconds): CPU ALLOCATION AND INITIALIZATION = %e (%2.3f %%total)\n",t1se,t1se/t0se*100.0);
    t2se = t2e-t2s;
    printf("DURATION 2 (seconds): CREATE PLAN AND STREAM = %e (%2.3f %%total)\n",t2se,t2se/t0se*100.0);
    t3se = t3e-t3s;
    printf("DURATION 3 (seconds): GPU ALLOCATION, COPY, AND INITIALIZATION = %e (%2.3f %%total)\n",t3se,t3se/t0se*100.0);
    t4se = t4e-t4s;
    printf("DURATION 4 (seconds): GPU CALCULATIONS (FFT and IFFT) = %e (%2.3f %%total)\n",t4se,t4se/t0se*100.0);
    t5se = t5e-t5s;
    printf("DURATION 5 (seconds): COPY REULT FROM GPU TO CPU = %e (%2.3f %%total)\n",t5se,t5se/t0se*100.0);
    printf("\n");
    

    //-----------------------------------------------------------
    return EXIT_SUCCESS;
}