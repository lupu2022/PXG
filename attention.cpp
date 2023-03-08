#include <chrono>

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include "tensortype/tensortype.hpp"
#include "attention.hpp"

void AttentionBlock::run(ncclComm_t comm) {
    tt::tensor_t x = tt::create_cuda_float( {SUB_BATCH, MAX_LENGTH, HIDDEN_SIZE} );
    tt::tensor_t w = tt::create_cuda_float( {HIDDEN_SIZE * 3, HIDDEN_SIZE} );
    tt::tensor_t b = tt::create_cuda_float( {HIDDEN_SIZE * 3} );

    tt::tensor_t y = tt::create_cuda_float( {SUB_BATCH, MAX_LENGTH, HIDDEN_SIZE * 3} );

    auto start = std::chrono::high_resolution_clock::now();
    x->op_linear(x, w, b, y);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "time: " << duration.count() << std::endl;
}
