#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

__device__ int taskIdx;         // "slateIdx"

// verify ID of SM
__device__ __inline__ uint32_t get_smid(){
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void TransformedKernel(  int sm_low, int sm_high,
                                    int* g_data, int inc_value,
                                    int grid_size,
                                    int *block_index,  int *max_blocks)
{
    __shared__ int smid;
    __shared__ bool valid;
    __shared__ int globIdx;

    __shared__ int logicalBlockIdx;
    __shared__ int physicalBlockIdx;
    
    
    if(threadIdx.x == 0){
        // logicalBlockIdx initialization
        logicalBlockIdx = 0;
        
        smid = get_smid();

        valid = !( smid < sm_low ||
                   smid > sm_high );
    }
    __syncthreads();
    
    if( !valid)
        return;

    int range = sm_high - sm_low + 1;           // number of PERSISTENT WORKERS
   
    if(threadIdx.x == 0)
    {
        physicalBlockIdx = atomicAdd(&(block_index[smid]), 1);
    }
    __syncthreads();


    while(1)
    {
        // check if there is enough place to host rest CTA
        while(physicalBlockIdx >= *max_blocks)
        {
            physicalBlockIdx = block_index[smid+1];         // wait for a moment , check again
        }

        if(threadIdx.x == 0)
        {
            globIdx = atomicAdd(&taskIdx, 1);
            
            logicalBlockIdx = globIdx + range;              // next index which PERSISTENT WORKER will have
        }

        __syncthreads();
        
        // original kernel
        int idx = globIdx * blockDim.x + threadIdx.x;
        g_data[idx] = g_data[idx] + inc_value;

        if(threadIdx.x == 0)
            atomicSub( &(block_index[smid]), 1);

        if(logicalBlockIdx >= grid_size)
            break;
    }

}


// original test kernel 
__global__ void incremental_kernel(int* g_data, int inc_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}


// check the correctness of kernel execution
bool correct_output( int *data, const int n, const int x)
{
    for(int i = 0; i < n; i++)
        if(data[i] !=x)
        {
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return false;
        }
    return true;
}
//////////////////////////////////////////////////////////////////////////////////////////
// Vector Addition Part
extern "C"
void scalarProdCPU(
        float *h_C,
        float *h_A,
        float *h_B,
        int vectorN,
        int elementN
        ){
    for(int vec = 0; vec < vectorN; vec++)
    {
        int vectorBase = elementN * vec;
        int vectorEnd = vectorBase + elementN;

        double sum = 0;

        for(int pos = vectorBase; pos < vectorEnd; pos++)
            sum += h_A[pos] * h_B[pos];

        h_C[vec] = (float)sum;
    }
}



/////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    printf("[%s] - Starting..\n", argv[0]);
    
    // parameter setting to use persistent thread
    int n = 60  * 1024;
    int nbytes = n * sizeof(int);
    int value = 10;

    const int num_sm = 30;
    
    // allocate host memory

    int *a = 0;                             // 1. input array
    cudaMallocHost((void**)&a, nbytes);
    memset(a, 0, nbytes);

    int host_max_blocks;                    // 2. set integer variable to memcpy to 'max_blocks'
    host_max_blocks = 5;

    // allocate device memory
    int *d_a=0;                             // 1. output array
    cudaMalloc((void**)&d_a, nbytes);
    cudaMemset(d_a, 255, nbytes);

    int *block_index = 0;                   // 2. SM usage reporting array
    cudaMalloc((void**)&block_index, sizeof(int) * (num_sm));
    cudaMemset(block_index, 0, sizeof(int) * (num_sm));

    int *max_blocks = 0;                    // 3. to let device know number of Maximum blocks that SM can host
    cudaMalloc((void**)&max_blocks, sizeof(int));
    cudaMemset(max_blocks, 0, sizeof(int));

    // set kernel launch configuration
    dim3 threads = dim3(1024, 1);
    dim3 blocks =  dim3( n/threads.x, 1);

    // create cuda event handles
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    float gpu_time = 0.0f;
    
    // original execution
    cudaEventRecord(start, 0);
    cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice);
    incremental_kernel<<<blocks, threads>>>(d_a, value);
    cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    // print the cpu and gpu times
    printf("time spent executing original kernel: %.2f\n", gpu_time);
  
    // check the output of correctness 
    bool bFinalResults = correct_output(a, n, value);
   
    ///////////////////////////////////////////////////////////////
    
    memset(a, 0, nbytes);
    cudaMemset(d_a, 255, nbytes);
    cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice);

    gpu_time = 0.0f;
    cudaEventRecord(start, 0);
    cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(max_blocks, &host_max_blocks, sizeof(int), cudaMemcpyHostToDevice);

    int currentIdx = 0;
    int start_sm = 6;
    int end_sm = 15;
    // dispatch part
    do {
        TransformedKernel<<<blocks, threads>>>( start_sm, end_sm,
                                                d_a, value,
                                                blocks.x,
                                               block_index, max_blocks);        
        cudaMemcpyFromSymbol(&currentIdx, taskIdx, sizeof(taskIdx), 0, cudaMemcpyDeviceToHost);
    }while( currentIdx < blocks.x);
                                           
    cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    printf("time spent executing second kernel: %.2f\n", gpu_time);
    
    // check the output for correctness
    bool bFinalResults2 = correct_output(a, n, value);

    // release resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(a);
    cudaFree(d_a);
    cudaFree(block_index);
    cudaFree(max_blocks);

    exit(bFinalResults2 ? EXIT_SUCCESS : EXIT_FAILURE);
}
