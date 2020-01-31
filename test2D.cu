#include <stdio.h>
#include <stdlib.h>

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
    __shared__ uint3 shared_blockID;
    
    const int leader = ( threadIdx.x == 0 &&
                         threadIdx.y == 0 &&
                         threadIdx.z == 0);

    if(leader){
        // logicalBlockIdx initialization
        logicalBlockIdx = 0;
        smid = get_smid();

        valid = !( smid < sm_low ||
                   smid > sm_high );
    }
    __syncthreads();
    
    if(!valid)
        return;

    int range = sm_high - sm_low + 1;               // number of PERSISTENT WORKERS
   
    if(leader)
    {
        physicalBlockIdx = atomicAdd(&(block_index[smid]), 1);
    }
    __syncthreads();
    

    while(1)
    {
        while(physicalBlockIdx >= *max_blocks)
        {
            physicalBlockIdx = block_index[smid];
        }

        if(leader)
        {
            globIdx = atomicAdd(&taskIdx, 1);
            
            logicalBlockIdx = globIdx + range;      // next index which PERSISTENT WORKER will have
            
            // calculate real block index which the user specified 
            //           from the modified index "globIdx" of transformed Grid K*
            shared_blockID.x = globIdx % gridDim.x -1;
            shared_blockID.y = globIdx / gridDim.x;
        }

        __syncthreads();
        
        // assume this case can have 1-D or 2-D grid like "slate"
        uint3 blockID = { shared_blockID.x, shared_blockID.y, 1 };
        
        // assume this case have 2D block
        int block_idx = blockID.y * gridDim.x + blockID.x;
        int thread_idx = block_idx * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
        
        // original kernel
        g_data[thread_idx] = g_data[thread_idx] + inc_value;

        if(leader)
            atomicSub( &(block_index[smid]),1);
        
        if(logicalBlockIdx >= grid_size)
            break;
    }
}


// original test kernel for 2D block
__global__ void incremental_kernel_2d(int* g_data, int inc_value)
{
    int block_idx  = blockIdx.y * gridDim.x + blockIdx.x;
    int thread_idx = block_idx * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    g_data[thread_idx] = g_data[thread_idx] + inc_value;
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

int main(int argc, char *argv[])
{
    printf("[%s] - Starting..\n", argv[0]);
    
    // parameter setting to use persistent thread
    //int n = 30  * 20 * 1024;              // i should handle overflow prob.
    int n = 30 * 20 * 16;   
    
    int nbytes = n * sizeof(int);
    int value = 10;

    const int num_sm = 30;
    
    // allocate host memory

    int *a = 0;                             // 1. input array
    cudaMallocHost((void**)&a, nbytes);
    memset(a, 0, nbytes);

    int host_max_blocks;                    // 2. set integer variable to memcpy to 'max_blocks'
    host_max_blocks = 5;

    int totalTask;                          // "slateMax" : total number of CTA

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
    //dim3 threads = dim3(32, 32 ,1);
    dim3 threads = dim3(4, 4, 1);
    dim3 blocks =  dim3(30 ,20, 1);
    
    totalTask = blocks.x * blocks.y * blocks.z;

    // create cuda event handles
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    float gpu_time = 0.0f;
    
    // original execution
    cudaEventRecord(start, 0);
    cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice);
    incremental_kernel_2d<<<blocks, threads>>>(d_a, value);
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
                                                totalTask,
                                                block_index, max_blocks);        
        cudaMemcpyFromSymbol(&currentIdx, taskIdx, sizeof(taskIdx), 0, cudaMemcpyDeviceToHost);
    }while( currentIdx < totalTask);
                                           
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
