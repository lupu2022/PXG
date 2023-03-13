#include <algorithm>

#include "cuda_tensor.hpp"
#include "kernels/kernels.h"

namespace tt {

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_dump(tensor_t self) {
    auto shape = shape_.vec();
    size_t first8 = std::min(shape.back(), (size_t)8);

    if ( DT == DataType::Float ) {
        auto stream = ComputingContext::cuda_stream;
        std::vector<float> local_first;
        std::vector<float> local_last;

        local_first.resize(first8, 0);
        local_last.resize(first8, 0);

        auto x = self->cuda_float();
        CUDA_CHECK(cudaMemcpyAsync(local_first.data(), x->data(), local_first.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));

        std::vector<size_t> pos = shape;
        for(int i = 0; i < pos.size() - 1; i++) {
            pos[i] = shape[i] - 1;
        }
        pos.back() = shape.back() - first8;
        void* src = (float *)x->data() + x->offset(pos);
        CUDA_CHECK(cudaMemcpyAsync(local_last.data(), src, local_last.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "--------------------------" << std::endl;
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << local_first[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << local_last[i] << " ";
        }
        std::cout << std::endl;
        return TT_OK;
    }

    return TT_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_zero(tensor_t self) {
    if ( DT == DataType::Float ) {

        void *dst = data();
        int n = shape_.numel();
        CUDA_CHECK( cudaMemset(dst, 0, n * sizeof(float)) );
        return TT_OK;
    }

    return TT_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_fill(tensor_t self, float value, size_t begin, size_t len) {
    if ( !ShapeType::is_dense(shape_, stride_) ) {
        return TT_TODO_ERROR;
    }

    if ( DT == DataType::Float ) {
        float* dst = (float *)data() + begin;
        std::vector<float> src;
        src.resize(len, value);

        CUBLAS_CHECK( cublasSetVector(len, sizeof(float), src.data(), 1, dst, 1) );
        return TT_OK;
    }

    return TT_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_linear(tensor_t self, tensor_t w_, tensor_t b_, tensor_t y_) {
    if ( DT == DataType::Float ) {
        auto x = this;
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

        kernels::LtSgemm(ComputingContext::cublasLt_handle,
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


/*
    batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
    return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
 */
#if 0
template<DataType DT>
std::variant<ComputingReturn, std::vector<tensor_t>> CUDATensor<DT>::op_split_qkv(tensor_t x_, int heads) {
    if ( DT == DataType::Float ) {
        auto x = x_->cuda_float();

        size_t batch = x->dims()[0];
        size_t tokens = x->dims()[1];
        size_t embedding3 = x->dims()[2];

        tt_assert( embedding3 % (heads * 3) == 0, "input's embedding size can't be splitted");
        size_t head_embedding = embedding3 / heads / 3;

        float* data = (float *)x->data();
        ShapeType qkv_shape( {batch, tokens, (size_t)heads, head_embedding} );
        size_t numel = batch * tokens * heads * head_embedding;

        std::vector<size_t> strides;
        strides.push_back(1);
        strides.push_back( head_embedding );
        strides.push_back( embedding3 );
        strides.push_back( tokens * embedding3 );

        std::vector<tensor_t> qkv;
        for(size_t i = 0; i < 3; i++) {
            void* src = data + numel * i;
            CUDATensor<DataType::Float>* cuda_tensor = new CUDATensor<DataType::Float>(src, qkv_shape, strides);
            tensor_t t = std::make_shared<TensorType>( cuda_tensor, qkv_shape );
            qkv.push_back(t);
        }
        return qkv;
    }
    return TT_TODO_ERROR;
}
#endif


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
