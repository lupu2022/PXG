#ifndef _PXG_ATTENTION_HPP_
#define _PXG_ATTENTION_HPP_

#include <vector>
#include <string>
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include "config.hpp"
#include "common.hpp"

#include "tensortype/tensortype.hpp"

struct CausalSelfAttention {
    CausalSelfAttention(const char* weights_file);
    CausalSelfAttention(std::vector<tensor_t>& weights);
    ~CausalSelfAttention();

    void zero_grad();
    std::vector<tensor_t> weights();
    std::vector<tensor_t> grads();

    tensor_t forward(tensor_t x);
    tensor_t backward(tensor_t _x);

private:
    create_local_tensors();

private:
    // from embedded to q,k,v
    tt::tensor_t    qkv_w_;
    tt::tensor_t    qkv_b_;

    tt::tensor_t    _qkv_w_;
    tt::tensor_t    _qkv_b_;

    // from embedded to embedded
    tt::tensor_t    out_w_;
    tt::tensor_t    out_b_;

    tt::tensor_t    _out_w_;
    tt::tesnor_t    _out_b_;
};

#endif
