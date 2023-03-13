#include "tensortype.hpp"
#include "cpu_tensor.hpp"
#include "cuda_tensor.hpp"

namespace tt {

int ComputingContext::cuda_device = -1;
cudaStream_t ComputingContext::cuda_stream = nullptr;
cublasHandle_t ComputingContext::cublas_handle = nullptr;
cublasLtHandle_t ComputingContext::cublasLt_handle = nullptr;
cudnnHandle_t ComputingContext::cudnn_handle = nullptr;
void* ComputingContext::cuda_workspace = nullptr;
size_t ComputingContext::cuda_workspace_size = 0;

void ComputingContext::boot(int cud) {
    cuda_device = cud;

    CUDA_CHECK( cudaSetDevice(cuda_device) );
    CUDA_CHECK( cudaStreamCreate(&cuda_stream) );

    CUBLAS_CHECK( cublasCreate_v2(&cublas_handle) );
    CUBLAS_CHECK( cublasLtCreate(&cublasLt_handle) );

    CUDNN_CHECK(cudnnCreate(&cudnn_handle));

    cuda_workspace_size = 1024 * 1024 * 32;
    CUDA_CHECK( cudaMalloc(&cuda_workspace, cuda_workspace_size) );
}

void ComputingContext::shutdown() {
    CUBLAS_CHECK( cublasLtDestroy(cublasLt_handle) );
    CUBLAS_CHECK( cublasDestroy(cublas_handle) );
    CUDA_CHECK( cudaStreamDestroy(cuda_stream) );
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

void TensorType::copy_to_cpu(tensor_t cpu_dst){

}

void TensorType::copy_from_cpu(tensor_t cpu_src){

}

}

