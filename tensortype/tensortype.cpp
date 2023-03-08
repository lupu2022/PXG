#include "tensortype.hpp"
#include "cpu_impl.hpp"
#include "cuda_impl.hpp"

namespace tt {

int ComputingContext::cuda_device = -1;
cublasHandle_t ComputingContext::cublas_handle = nullptr;
cublasLtHandle_t ComputingContext::cublasLt_handle = nullptr;
void* ComputingContext::cuda_workspace = nullptr;
size_t ComputingContext::cuda_workspace_size = 0;

void ComputingContext::init(int cud, cublasHandle_t cubh, cublasLtHandle_t clth) {
    cuda_device = cud;
    cublas_handle = cubh;
    cublasLt_handle = clth;

    cuda_workspace_size = 1024 * 1024 * 32;
    CUDA_CHECK( cudaMalloc(&cuda_workspace, cuda_workspace_size) );
}


TensorType::~TensorType() {
    if ( impl_index() == ImplType::CUDA_FLOAT ) {
        cuda_float_t* tensor = std::get<CUDA_FLOAT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CUDA_BF16 ) {
        cuda_bf16_t* tensor = std::get<CUDA_BF16>(impl_);
        delete tensor;
    }
}

TransformerComputing* TensorType::impl() {
    if ( impl_index() == ImplType::CUDA_FLOAT ) {
        cuda_float_t* tensor = std::get<CUDA_FLOAT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CUDA_BF16 ) {
        cuda_bf16_t* tensor = std::get<CUDA_BF16>(impl_);
        return tensor;
    }

    tt_panic("Can't be here!");
    return nullptr;
}

}

