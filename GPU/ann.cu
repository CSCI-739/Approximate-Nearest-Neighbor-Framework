#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__ void findNearestNeighborCosine(float *points, float *queries, float *max_cosine, int *max_index, int n, int num_queries, int dimensions, float target_similarity) {
    extern __shared__ char shared[];
    float *s_cosine = (float*)shared;
    int *s_index = (int*)(shared + blockDim.x * sizeof(float));

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int qid = blockIdx.y;

    if (tid < n && qid < num_queries) {
        float dot_product = 0, query_magnitude = 0, point_magnitude = 0;
        for (int d = 0; d < dimensions; ++d) {
            int idx = tid * dimensions + d;
            int q_idx = qid * dimensions + d;
            dot_product += queries[q_idx] * points[idx];
            query_magnitude += queries[q_idx] * queries[q_idx];
            point_magnitude += points[idx] * points[idx];
        }
        query_magnitude = sqrt(query_magnitude);
        point_magnitude = sqrt(point_magnitude);

        float cosine_similarity = 0;
        if (query_magnitude > 0 && point_magnitude > 0) {
            cosine_similarity = dot_product / (query_magnitude * point_magnitude);
        }

        s_cosine[threadIdx.x] = cosine_similarity;
        s_index[threadIdx.x] = tid;
        if(cosine_similarity > target_similarity)
            max_index[qid] = tid;
        __syncthreads();
    }
}


std::vector<std::vector<float>> read_matrix(FILE* fin, int row, int col) {
    std::vector<std::vector<float>> ret;
    for (int i = 0; i < row; ++i) {
        std::vector<float> curr;
        float tmp = 0;
        for (int j = 0; j < col; ++j) {
            fscanf(fin, "%f", &tmp);
            curr.push_back(tmp);
        }
        ret.push_back(curr);
    }
    return ret;
}

int main(int argc, char* argv[]) {
    FILE* fin = fopen(argv[1], "r");
    FILE* fout = fopen(argv[2], "w");

    int n = 0, d = 0, m = 0;
    float target_similarity = 0.9;
    fscanf(fin, "%d%d%d", &d, &n, &m);

    std::vector<std::vector<float>> base = read_matrix(fin, n, d);
    std::vector<std::vector<float>> query = read_matrix(fin, m, d);

    float* flat_base = new float[n * d];
    float* flat_query = new float[m * d];
    for (int i = 0; i < n; ++i)
        memcpy(flat_base + i * d, base[i].data(), d * sizeof(float));
    for (int i = 0; i < m; ++i)
        memcpy(flat_query + i * d, query[i].data(), d * sizeof(float));

    float* d_base, * d_query, *d_max_cosine;
    int *d_max_index;

    cudaMalloc(&d_base, n * d * sizeof(float));
    cudaMalloc(&d_query, m * d * sizeof(float));
    cudaMalloc(&d_max_cosine, m * sizeof(float));
    cudaMalloc(&d_max_index, m * sizeof(int));

    float *max_cosine_host = new float[m];
    for (int i = 0; i < m; i++) {
        max_cosine_host[i] = -1.0f;
    }

    cudaMemcpy(d_base, flat_base, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, flat_query, m * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_cosine, max_cosine_host, m * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, m);

    int sharedMemSize = threadsPerBlock.x * (sizeof(float) + sizeof(int));
    findNearestNeighborCosine<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_base, d_query, d_max_cosine, d_max_index, n, m, d, target_similarity);

    int *max_index_host = new int[m];
    cudaMemcpy(max_cosine_host, d_max_cosine, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_index_host, d_max_index, m * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; ++i) {
        fprintf(fout, "%d\n", max_index_host[i]);
    }

    cudaFree(d_base);
    cudaFree(d_query);
    cudaFree(d_max_cosine);
    cudaFree(d_max_index);

    return 0;
}