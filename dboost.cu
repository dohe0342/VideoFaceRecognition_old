#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/container_utils.hpp>
#include <boost/python/numpy.hpp>
#include <stdio.h>
#include <vector>

using namespace std;
namespace py = boost::python;

float* golden_gpu;
float* frame_gpu;
float* result_gpu;
float* reduced_gpu;
float* reduced_index_gpu;
float* reduced_cpu;
float* reduced_index_cpu;

__device__ const int TILE_WIDTH = 16;


__global__ 
void matmul(float *result, float *frame, float *golden, int m, int n){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = tid / n;
    const int col = tid % n;
    if (tid < m * n){
        result[tid] = 0;
        for(int i = 0; i < 512; ++i){
            result[tid] += frame[row * 512 + i] * golden[i * n + col];
        }
    }
}

__global__
void matmul_tiled(float* result, float* frame, float* golden, int n, int offset){

    __shared__ float frame_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float golden_tile[TILE_WIDTH][TILE_WIDTH];

    int blkIdx = blockIdx.x + offset * 65535;

    int res_trow = blkIdx / (n / TILE_WIDTH), res_tcol = blkIdx % (n / TILE_WIDTH);
    int in_row = threadIdx.x / TILE_WIDTH, in_col = threadIdx.x % TILE_WIDTH;
    
    float res = 0;

    for(int i = 0; i < 512 / TILE_WIDTH; ++i){
        frame_tile[in_row][in_col] = frame[512 * (res_trow * TILE_WIDTH + in_row) + i * TILE_WIDTH + in_col];
        golden_tile[in_row][in_col] = golden[n * (i * TILE_WIDTH + in_row) + res_tcol * TILE_WIDTH + in_col];

        __syncthreads();

        for(int j = 0; j < TILE_WIDTH; j++)
            res += frame_tile[in_row][j] * golden_tile[j][in_col];

        __syncthreads();
    }
    result[n * (res_trow * TILE_WIDTH + in_row) + res_tcol * TILE_WIDTH + in_col] = res;
}


//devblogs.nvidia.com/faster-parallel-reductions-kepler/
__inline__ __device__
float warpReduceMax(float val){
    for(int offset = warpSize/2; offset > 0; offset /= 2)
        val = max(val, __shfl_down(val, offset));
    return val;
}


__inline__ __device__
float blockReduceMax(float val){
    static __shared__ float shared[32]; // 32 partitial sum
    int lane = threadIdx.x % warpSize; // lane in warp
    int wid = threadIdx.x / warpSize; // warp id

    val = warpReduceMax(val); // Each warp performs reduction
    
    if (lane == 0) shared[wid] = val; // Write reduced value
    
    __syncthreads();

    // read only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize)
        ? shared[lane] : 0;

    if(wid == 0) val = warpReduceMax(val);

    return val;
}


// reduce accross a complete grid(may thread blocks).
__global__ void maxReduce(float* in, float* out, int N){
    float res = 0;
    //Reduce multiple elements per thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tid; i < N; i += stride){
            res = max(res, in[i]);
    } // grid stride loop, res = blockDim stride max
    res = blockReduceMax(res);
    if(threadIdx.x == 0)
        out[blockIdx.x] = res;
}

__global__ void findIndex(float* res, float* maxArr, float* out, int M, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid / N;
    int idx = tid % N;
    if(tid < M * N && res[tid] == maxArr[row])
        out[row] = idx;
}


void deviceReduce(float* in, float* out, int N){
    int threads = 512;
    int blocks = min((N + threads - 1) / threads, 1024);

    maxReduce<<<blocks, threads>>>(in, out, N);
    maxReduce<<<1, 1024>>>(out, out, blocks);
}


vector<float> launch_kernel(int m, int n, int tiled){
    vector<float> result(2 * m);

    int i = 0;

    if(tiled){
        int augm = TILE_WIDTH * (m / TILE_WIDTH + 1);
        for(i = 0; (i + 1) * 65535  < augm * n / (TILE_WIDTH * TILE_WIDTH); i++)
            matmul_tiled<<<65535, TILE_WIDTH * TILE_WIDTH>>>(result_gpu, frame_gpu, golden_gpu, n, i);
        matmul_tiled<<<augm * n / (TILE_WIDTH * TILE_WIDTH) - 65535 * i, TILE_WIDTH * TILE_WIDTH>>>
            (result_gpu, frame_gpu, golden_gpu, n, i);
    }
    else
        matmul<<<(m * n / 512) + 1, 512>>>(result_gpu, frame_gpu, golden_gpu, m, n);

    for(int i = 0; i < m; ++i){
        deviceReduce(result_gpu + i * n, reduced_gpu + i, n);
    }

    findIndex<<<(m * n / 512) + 1, 512>>>(result_gpu, reduced_gpu, reduced_index_gpu, m, n);

    cudaError_t stat1 = cudaMemcpy(reduced_cpu, reduced_gpu, sizeof(float) * m, cudaMemcpyDeviceToHost);
    cudaError_t stat2 = cudaMemcpy(reduced_index_cpu, reduced_index_gpu, sizeof(float) * m, cudaMemcpyDeviceToHost);

    if(stat1 != cudaSuccess || stat2 != cudaSuccess)
        perror("cudaMemcpy error");

    for(int i = 0; i < m; ++i){
        result[2 * i] = reduced_index_cpu[i];
        result[2 * i + 1] = reduced_cpu[i];
    }

/*
    for(int i = 0; i < m; ++i){
        float maxSim = -999;
        for(int j = 0; j < n; ++j){
            if(maxSim < result_cpu[i * n + j]){
                result[2 * i] = j;
                result[2 * i + 1] = result_cpu[i * n + j];
                maxSim = result_cpu[i * n + j];
            }
        }
    }
*/

    return result;
}


extern "C"
void golden_init(int amt){
//    printf("golden amt %d\n", amt);
    size_t size = amt * 512 * sizeof(float);
    cudaMalloc(&golden_gpu, size);
    cudaMemset(golden_gpu, 0, size);
}


extern "C"
void golden_h2d(int amt, float* cpu){
//    printf("copied sample of b_cpu\n");
//    for(int i = 0; i < 10; ++i)
//        printf("%f\n", cpu[i]);
    size_t size = amt * 512 * sizeof(float);
    cudaError_t stat = cudaMemcpy(golden_gpu, cpu, size, cudaMemcpyHostToDevice); 
    if(stat != cudaSuccess){
        perror("CUDA MEMCPY ERROR\n");
        exit(1);
    }
}


extern "C"
void golden_free(){
    cudaFree(golden_gpu);
}


extern "C"
void frame_init(int amt){
//    printf("frame amt %d\n", amt);
    size_t size = amt * 512 * sizeof(float);
    cudaMalloc(&reduced_gpu, size / 512);
    cudaMalloc(&reduced_index_gpu, size / 512);
    cudaMalloc(&frame_gpu, size);
    cudaMemset(frame_gpu, 0, size);
}


extern "C"
void frame_h2d(int amt, float* cpu){
    size_t size = amt * 512 * sizeof(float);
    cudaError_t stat = cudaMemcpy(frame_gpu, cpu, size, cudaMemcpyHostToDevice); 
    if(stat != cudaSuccess){
        perror("CUDA MEMCPY ERROR\n");
        exit(1);
    }
}


extern "C"
void frame_free(){
    cudaFree(frame_gpu);
}


extern "C"
void result_init(int amt){
    size_t size = amt * sizeof(float);
    reduced_cpu = (float*)malloc(size);
    reduced_index_cpu = (float*)malloc(size);
    cudaMalloc(&result_gpu, size);
}


extern "C"
void result_free(){
    cudaFree(result_gpu);
    free(reduced_cpu);
    free(reduced_index_cpu);
}


BOOST_PYTHON_MODULE(dboost){
    Py_Initialize();
    
    py::class_<std::vector<float> >("FloatVec")
        .def(py::vector_indexing_suite<std::vector<float> >());
    py::def("launch_kernel", launch_kernel);
    py::def("golden_init", golden_init);
    py::def("golden_h2d", golden_h2d);
    py::def("golden_free", golden_free);
    py::def("frame_init", frame_init);
    py::def("frame_h2d", frame_h2d);
    py::def("frame_free", frame_free);
    py::def("result_init", result_init);
    py::def("result_free", result_free);
}

