#ifndef _PXG_ATTENTION_HPP_
#define _PXG_ATTENTION_HPP_

#include <vector>
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include "config.hpp"
#include "common.hpp"

struct DeviceContext {
    DeviceContext(int device,int rank, int ranks) : rank_(rank), ranks_(ranks), cuda_device_(device) {
        CUDACHECK( cudaSetDevice(cuda_device_) );
        CUDACHECK( cudaStreamCreate(&cuda_stream_) );
        CUBLASCHECK( cublasCreate_v2(&cublas_handle_) );
        CUBLASCHECK( cublasLtCreate(&cublasLt_handle_) );
    }
    ~DeviceContext() {
        if ( cublas_handle_ != nullptr ) {
            CUBLASCHECK( cublasDestroy_v2(cublas_handle_) );
        }
    }

    const int rank_;
    const int ranks_;
    const int cuda_device_;
    cublasHandle_t cublas_handle_;
    cublasLtHandle_t cublasLt_handle_;
    cudaStream_t cuda_stream_;
};

struct AttentionBlock {
    AttentionBlock(DeviceContext& ctx) : ctx_(ctx) { }
    ~AttentionBlock() { }
    void run(ncclComm_t comm);

private:
    DeviceContext ctx_;
};

#endif
