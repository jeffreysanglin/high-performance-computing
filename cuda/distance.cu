#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <float.h>
#include <cmath>
#include "knn_ref.cpp"

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
#define K 15

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                    << " at " << __FILE__ << ":" << __LINE__    \
                    << std::endl;                               \
        std::exit(1);                                           \
    }                                                           \
} while(0)

__global__ void compute_norms(
    const float* A,
    float* norms,
    const int rows,
    const int dims
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; //1D kernel, so all x's.
    if (row >= rows) return;
    float norm = 0.0f;

    //Calculate our partial norm for A -- summation(x^2)
    for (int i = 0; i < dims; ++i){
        float val = A[row * dims + i]; //thread can access all of data, so we need to specify which element
        norm += val * val;
    }
    // Sqr root the row output.
    norms[row] = sqrtf(norm);
}

__global__ void cos_sim_tiled_kernel(
    const float* X, const float* Y, float* D,
    const float* Xnorms, const float* Ynorms,
    float* Scores, int* Indices,
    int rows, int dims, int k_neighbors) 
{
    __shared__ float Xs[TILE_SIZE][TILE_SIZE];
    __shared__ float Ys[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    //Compute the cosine similarity
    //Matmul
    float sum = 0.0f;
    int numTiles = (dims + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int xDim = t * TILE_SIZE + threadIdx.x;
        int yDim = t * TILE_SIZE + threadIdx.y;

        Xs[threadIdx.y][threadIdx.x] = 
            (row < rows && xDim < dims) 
            ? X[row * dims + xDim] 
            : 0.0f;
        //Since we are transposing Y, the output column is a Y row.
        Ys[threadIdx.x][threadIdx.y] = 
            (col < rows && yDim < dims) 
            ? Y[col * dims + yDim] 
            : 0.0f;

        __syncthreads();

        for (int tl = 0; tl < TILE_SIZE; ++tl) {

            //CUDA treats "y" as "rows" and "x" as "columns".
            //We can follow our same format as above and sum the terms together
            // ... using sum and k
            //sum += A[row * N + k] * B[k * N + col];
            sum += Xs[threadIdx.y][tl] * Ys[threadIdx.x][tl]; //transpose the mul for Y
        }

        __syncthreads();
    }

    //Denominator and Calc D
    if (row < rows && col < rows) {
        float denom = Xnorms[row] * Ynorms[col];
        D[row * rows + col] = (denom == 0.0f) ? 0.0f : sum / denom;
    }    
}

__device__ void insert_topk(
    float score, 
    int index, 
    float best_scores[K], 
    int best_indices[K]
)
{
    //If the score on the right is smaller than the score on left, 
    //  then we don't need to change anything.
    if (score <= best_scores[K - 1]) {
        return;
    }

    int pos = K - 1;

    // If we know that the current score is better than two slots to left, 
    // then move it.
    while (pos > 0 && score > best_scores[pos - 1]) {
        best_scores[pos] = best_scores[pos - 1];
        best_indices[pos] = best_indices[pos - 1];
        // Then continue moving our position to the left.
        --pos; // pos = pos - 1;
        // Close out the loop when our position reaches 0.
    }

    best_scores[pos] = score;
    best_indices[pos] = index;

}

__global__ void knn_tiled_kernel(
    const float* X, const float* Y, float* D,
    const float* Xnorms, const float* Ynorms,
    float* TopScores, int* TopIndices,
    const int rows, const int dims, const int k_neighbors) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) {
        return;
    }

    float best_scores[K];
    int best_indices[K];
    // Fill our temp arrays
    for (int i = 0; i < K; ++i) {
        best_scores[i] = -FLT_MAX;
        best_indices[i] = -1;
    }
    // Grab the highest values and store them in temp arrays
    for (int col = 0; col < rows; ++col) {
        float score = D[row * rows + col];
        insert_topk(score, col, best_scores, best_indices);      
    }

    //Only pass the best values to the TopScores, TopIndices
    for (int i = 0; i < K; ++i) {
        TopScores[row * K + i] = best_scores[i];
        TopIndices[row * K + i] = best_indices[i];
    }    
}

void knn_cuda_tiled(
    const float* h_X, const float* h_Y, float* h_D,
    float* h_Scrs, int* h_Idxs,
    int rows, int dims, int k,
    const int block_size, const int measure) 
{
    float *d_X, *d_Y, *d_Scrs, *d_D;
    float *d_Xnorms, *d_Ynorms;
    int *d_Idxs;
    size_t data_bytes = rows * dims * sizeof(float);
    size_t dist_bytes = rows * rows * sizeof(float);
    size_t score_bytes = rows * k * sizeof(float);
    size_t index_bytes = rows * k * sizeof(int);
    size_t norm_bytes = rows * sizeof(float);

    //Mem Allocation - X,Y
    CUDA_CHECK(cudaMalloc(&d_X, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_Y, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_D, dist_bytes));
    CUDA_CHECK(cudaMalloc(&d_Xnorms, norm_bytes));
    CUDA_CHECK(cudaMalloc(&d_Ynorms, norm_bytes));
    
    //Mem Allocation, Outputs - Scrs, Idxs
    CUDA_CHECK(cudaMalloc(&d_Scrs, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_Idxs, index_bytes));
    
    //Copy data to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, data_bytes, cudaMemcpyHostToDevice));

    //Block size - based on the work the kernel will do: 
        //distribute rows and compute them
    dim3 threads_per_block_2D(TILE_SIZE, TILE_SIZE);
    dim3 total_blocks_2D(
        (rows + TILE_SIZE - 1) / TILE_SIZE,
        (rows + TILE_SIZE - 1) / TILE_SIZE
    );

    //norms
    int threads_per_block_1D = block_size;
    int total_blocks_1D = (rows + threads_per_block_1D - 1) / threads_per_block_1D;


    // Set up our timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    //The norm kernel
    compute_norms<<<total_blocks_1D, threads_per_block_1D>>>(
        d_X,
        d_Xnorms,
        rows,
        dims
    );
    CUDA_CHECK(cudaGetLastError());
    compute_norms<<<total_blocks_1D, threads_per_block_1D>>>(
        d_Y,
        d_Ynorms,
        rows,
        dims
    );
    CUDA_CHECK(cudaGetLastError());

    //Computing Cosine Similarity
    cos_sim_tiled_kernel<<<total_blocks_2D, threads_per_block_2D>>>(
        d_X,
        d_Y,
        d_D,
        d_Xnorms,
        d_Ynorms,
        d_Scrs,
        d_Idxs,
        rows,
        dims,
        k
    );
    CUDA_CHECK(cudaGetLastError());

    //Grabbing highest scores
    knn_tiled_kernel<<<total_blocks_1D, threads_per_block_1D>>>(
        d_X,
        d_Y,
        d_D,
        d_Xnorms,
        d_Ynorms,
        d_Scrs,
        d_Idxs,
        rows,
        dims,
        k
    );
    CUDA_CHECK(cudaGetLastError());


    //Record the stop
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double micro_sec = ms * 1e3;

    if (measure == 1) {
        std::cout << "CUDA Tiled KNN: "
                    << micro_sec << " us\n";
    }

    //Pull from device to host
    CUDA_CHECK(cudaMemcpy(h_Scrs, d_Scrs, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Idxs, d_Idxs, index_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D, d_D, dist_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_Xnorms));
    CUDA_CHECK(cudaFree(d_Ynorms));
    CUDA_CHECK(cudaFree(d_Scrs));
    CUDA_CHECK(cudaFree(d_Idxs));
    CUDA_CHECK(cudaEventDestroy(start)); 
    CUDA_CHECK(cudaEventDestroy(stop));
}


int main(int argc, char** argv) {
    int seed = 42;
    srand(seed);

    int block_size = 256;
    int n_rows = 1000;
    int n_dims = 256;
    
    if (argc > 1) block_size = std::atoi(argv[1]);
    if (argc > 2) n_rows = std::atoi(argv[2]);
    if (argc > 3) n_dims = std::atoi(argv[3]);

    std::cout << "block_size: " << block_size << "\n";
    std::cout << "rows:       " << n_rows << "\n";
    std::cout << "dims:       " << n_dims << "\n";

    //Length of array set
    //Dimensions
    //Number of neighbors promoted to constant Define above.
    //int n_neighbors = K;

    int N = n_rows * n_dims; // replace with actual N
    // if (argc > 1) N = std::atoi(argv[1]);

    std::vector<float> X(N), Y(N), D(n_rows * n_rows);
    // std::vector<float> D(rows * rows);
    std::vector<float> Scrs(n_rows * K);
    std::vector<int> Idxs(n_rows * K);

    // initialize X, Y
    for (int i = 0; i < N; ++i) {
        X[i] = static_cast<float>(rand())/RAND_MAX;
        Y[i] = static_cast<float>(rand())/RAND_MAX;
    }

    // warm-up calls
    knn_cuda_tiled(
        X.data(), Y.data(), D.data(),
        Scrs.data(), Idxs.data(),
        n_rows, n_dims, K,
        block_size, 0
    );

    // Take measurements
    knn_cuda_tiled(
        X.data(), Y.data(), D.data(),
        Scrs.data(), Idxs.data(),
        n_rows, n_dims, K,
        block_size, 1 //Yes, measure
    );

    std::vector<float> RefScrs(n_rows * K);
    std::vector<int> RefIdxs(n_rows * K);

    knn_serial(
        X, Y, 
        RefScrs, RefIdxs, 
        n_rows, n_dims
    );
    
    float tolerance = 1e-4f;
    int mismatches = 0;

    for (int i = 0; i < n_rows * K; ++i) {
        float diff = std::fabs(Scrs[i] - RefScrs[i]);
        //Only count as miss if they are dramatically different.
        if (diff > tolerance) {
            if (mismatches < 10) {
                std::cout
                    << "Mismatch at " << i
                    << ": GPU idx=" << Idxs[i]
                    << ", CPU idx=" << RefIdxs[i]
                    << ", GPU score=" << Scrs[i]
                    << ", CPU score=" << RefScrs[i]
                    << ", diff=" << diff
                    << "\n";
            }

            ++mismatches;
        }
    }

    if (mismatches == 0) {
        std::cout << "Correctness Check PASSED.\n";
    } else {
        std::cout << "Correctness Check FAILED. Mis-mathes: "
                    << mismatches << "\n";
    }
   
    return 0;

}