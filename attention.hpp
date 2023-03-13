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
    CausalSelfAttention(std::vector<tt::tensor_t>& weights);
    ~CausalSelfAttention();

    void zero_grad();
    std::vector<tt::tensor_t> weights();
    std::vector<tt::tensor_t> grads();

    tt::tensor_t forward(tt::tensor_t x);
    tt::tensor_t backward(tt::tensor_t _x);

    void test();
private:
    void create_local_tensors();

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
    tt::tensor_t    _out_b_;
};

#endif
