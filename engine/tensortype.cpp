#include "tensortype.hpp"
#include "cpu_impl.hpp"
#include "cuda_impl.hpp"

namespace tt {

int ComputingContext::cuda_device = -1;
cublasHandle_t ComputingContext::cublas_handle = nullptr;
void ComputingContext::init(int cud, cublasHandle_t cubh) {
    cuda_device = cud;
    cublas_handle = cubh;
}


TensorType::~TensorType() {

}

TransformerComputing* TensorType::impl() {
    return nullptr;
}

}

