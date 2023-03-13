#ifndef _CUDA_IMPL_HPP_
#define _CUDA_IMPL_HPP_

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

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

#define CUDNN_CHECK(f) \
    do { \
        cudnnStatus_t  s_ = f; \
        if (s_ != CUDNN_STATUS_SUCCESS) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

namespace tt {

template <DataType _DTYPE_>
struct CUDATensor : public TransformerComputing {
    ~CUDATensor() {
        if (mem_ != nullptr && owner_) {
            CUDA_CHECK(cudaFree(mem_));
        }
    }

    CUDATensor(const ShapeType& shape) : shape_(shape), stride_(shape.dense_strides()), owner_(true) {
        tt_assert(shape.vec().size() != 0, "Can't build tensor with zero shape!");
        CUDA_CHECK(cudaMalloc(&mem_, shape_.numel() * DataType_size(_DTYPE_)));
    }

    CUDATensor(void *mem, const ShapeType& shape, const std::vector<size_t> stride) : mem_(mem), shape_(shape), stride_(stride), owner_(false) {
        tt_assert(shape.vec().size() != 0, "Can't build tensor with zero shape!");
    }

    void* data() {
        return mem_;
    }
    const std::vector<size_t>& dims() {
        return shape_.vec();
    }
    const ShapeType& shape() {
        return shape_;
    }
    const std::vector<size_t>& stride() {
        return stride_;
    }

    cudnnTensorDescriptor_t create_cudnn_td_with(const std::vector<size_t> shape) {
        cudnnDataType_t dtype;
        cudnnTensorFormat_t  format = CUDNN_TENSOR_NCHW;
        cudnnTensorDescriptor_t desc;

        if ( _DTYPE_ == DataType::Float ) {
            dtype = CUDNN_DATA_FLOAT;
        } else if ( _DTYPE_ == DataType::BF16 ) {
            dtype = CUDNN_DATA_BFLOAT16;
        } else {
            tt_panic("cudnn don't support!");
        }

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

        if (shape.size() == 4) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], shape[1], shape[2], shape[3]));
        } else if (shape.size() == 3) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], shape[1], 1, shape[2]));
        } else if (shape.size() == 2) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], shape[1], 1, 1));
        } else if (shape.size() == 1) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, 1, shape[0], 1, 1));
        } else {
            tt_panic("cudnnSetTensor4dDescriptor: can't convert shape");
        }

        return desc;
    }

    cudnnTensorDescriptor_t create_cudnn_td() {
        return create_cudnn_td_with( shape_.vec() );
    }

    virtual ComputingReturn op_linear(tensor_t w, tensor_t b, tensor_t y);

private:
    void*                       mem_;
    ShapeType                   shape_;
    const std::vector<size_t>   stride_;
    const bool                  owner_;
};


} // end of namespace
#endif
