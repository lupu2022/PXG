#ifndef _CUDA_IMPL_HPP_
#define _CUDA_IMPL_HPP_

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "tensortype.hpp"

#define COMPLAIN_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns error: %d.\n", __FILE__, __LINE__, \
                what, status); \
        exit(1); \
    } while (0)

#define CUDA_CHECK(f) \
    do { \
        cudaError_t  s_ = f; \
        if (s_ != cudaSuccess) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define CUBLAS_CHECK(f) \
    do { \
        cublasStatus_t  s_ = f; \
        if (s_ != CUBLAS_STATUS_SUCCESS) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

namespace tt {

// some helping stuff
extern int cuda_device;
extern cublasHandle_t cublas_handle;

template <DataType _DTYPE_>
struct CUDATensor : public TransformerComputing {
    ~CUDATensor() {
        if (mem_ != nullptr) {
            CUDA_CHECK(cudaFree(mem_));
        }
    }

    CUDATensor(const ShapeType& shape) : shape_(shape.vec()), stride_(shape.dense_strides()) {
        tt_assert(shape.vec().size() != 0, "Can't build tensor with zero shape!");
    }

private:
    void*                       mem_;
    const std::vector<size_t>   shape_;
    const std::vector<size_t>   stride_;
};

} // end of namespace
#endif