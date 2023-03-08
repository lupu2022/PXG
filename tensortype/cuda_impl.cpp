#include "cuda_impl.hpp"
#include "kernels/LtSgemm.h"

namespace tt {

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_linear(tensor_t x_, tensor_t w_, tensor_t b_, tensor_t y_) {
    if ( DT == DataType::Float ) {
        auto x = x_->cuda_float();
        auto w = w_->cuda_float();
        auto b = b_->cuda_float();
        auto y = y_->cuda_float();

        size_t batch = x->dims()[0];
        size_t tokens = x->dims()[1];
        size_t inSize = x->dims()[2];
        size_t outSize = w->dims()[0];

        int m = outSize;
        int n = batch * tokens;
        int k = inSize;

        float* A = (float *)w->data();
        float* B = (float *)x->data();
        float* C = (float *)y->data();

        float alpha = 1.0;
        float beta = 0.0;

        LtSgemm(ComputingContext::cublasLt_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, k,
                B, k, &beta,
                C, m,
                ComputingContext::cuda_workspace,
                ComputingContext::cuda_workspace_size);

        return TT_OK;
    }

    return TT_TODO_ERROR;
}

tensor_t create_cuda_float(std::vector<size_t> shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::Float>* tensor = new CUDATensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cuda_bf16(std::vector<size_t> shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::BF16>* tensor = new CUDATensor<DataType::BF16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

}
