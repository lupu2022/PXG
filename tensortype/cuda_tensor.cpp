#include "cuda_tensor.hpp"
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
        void* bias = b->data();

        float alpha = 1.0;
        float beta = 0.0;

        /*
        auto stream = ComputingContext::cuda_stream;
        std::vector<float> localA;
        localA.resize(inSize, 0.1);

        std::vector<float> localB;
        localB.resize(inSize, 3.14);

        std::vector<float> localBias;
        localBias.resize(inSize, 1000.0);

        CUDA_CHECK( cudaMemcpyAsync(A, localA.data(), localA.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK( cudaMemcpyAsync(B, localB.data(), localB.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK( cudaMemcpyAsync(bias, localBias.data(), localBias.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        */

        LtSgemm(ComputingContext::cublasLt_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, k,
                B, k, &beta,
                C, m,
                ComputingContext::cuda_workspace,
                ComputingContext::cuda_workspace_size);

        {
            auto ydesc = y->create_cudnn_td_with({batch, 1, tokens, outSize});
            auto bdesc = b->create_cudnn_td_with({1, 1, 1, outSize});

            beta = 1.0;
            CUDNN_CHECK( cudnnAddTensor(ComputingContext::cudnn_handle,
                                        &alpha, bdesc, bias,
                                        &beta, ydesc, C));
        }

        /*
        CUDA_CHECK(cudaMemcpyAsync(localA.data(), C, localA.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cout << " ##################### " <<  localA[0] << std::endl;
        */

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
