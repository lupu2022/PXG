#ifndef _PXG_ATTENTION_HPP_
#define _PXG_ATTENTION_HPP_

#include <vector>
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include "config.hpp"
#include "common.hpp"

struct AttentionBlock {
    AttentionBlock()  { }
    ~AttentionBlock() { }
    void run(ncclComm_t comm);
};

#endif
