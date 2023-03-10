#include <chrono>

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include "config.hpp"
#include "attention.hpp"

/*
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
*/

CausalSelfAttention::CausalSelfAttention(const char* weights_file) {
    create_local_tensors();
}

CausalSelfAttention::CausalSelfAttention(std::vector<tt::tensor_t>& weights) {
    create_local_tensors();
}

void CausalSelfAttention::create_local_tensors() {
    // create weight tensors
    qkv_w_ =  tt::create_cuda_float( {3, HIDDEN_SIZE, HIDDEN_SIZE} );
    qkv_b_ = tt::create_cuda_float( {3, HIDDEN_SIZE} );

    out_w_ = tt::create_cuda_float( {HIDDEN_SIZE, HIDDEN_SIZE} );
    out_b_ = tt::create_cuda_float( {HIDDEN_SIZE} );

    // create gradient tensors
    _qkv_w_ =  tt::create_cuda_float( {3, HIDDEN_SIZE, HIDDEN_SIZE} );
    _qkv_b_ = tt::create_cuda_float( {3, HIDDEN_SIZE} );

    _out_w_ = tt::create_cuda_float( {HIDDEN_SIZE, HIDDEN_SIZE} );
    _out_b_ = tt::create_cuda_float( {HIDDEN_SIZE} );
}

void CausalSelfAttention::zero_grad() {
    qkv_w_->op_zero();
    qkv_b_->op_zero();
    out_w_->op_zero();
    out_b_->op_zero();
}

std::vector<tt::tensor_t> CausalSelfAttention::weights() {
    std::vector<tt::tensor_t> ret{qkv_w_, qkv_b_, out_w_, out_b_};
    return ret;
}

std::vector<tt::tensor_t> CausalSelfAttention::grads() {
    std::vector<tt::tensor_t> ret{_qkv_w_, _qkv_b_, _out_w_, _out_b_};
    return ret;
}

tt::tensor_t CausalSelfAttention::forward(tt::tensor_t x) {
    return x;
}

tt::tensor_t CausalSelfAttention::backward(tt::tensor_t grad) {
    return grad;
}


