#ifndef _TENSORTYPE_HPP_
#define _TENSORTYPE_HPP_

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cudnn.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include <variant>

#define tt_assert(Expr, Msg) \
    tt::_M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)

#define tt_panic(Msg) \
    tt::_M_Panic(__FILE__, __LINE__, Msg)

#define tt_check(ret, Msg)                     \
    if ( ret != tt::TT_OK ) {                  \
        tt::_M_Panic(__FILE__, __LINE__, Msg); \
    }                                          \
    return ret

namespace tt {
// some common help functions
inline void _M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg) {
    if (!expr) {
        std::cerr << "TT::Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

inline void _M_Panic(const char* file, int line, const char* msg) {
    std::cerr << "TT::Panic:\t" << msg << "\n"
        << "Source:\t\t" << file << ", line " << line << "\n";
    abort();
}


enum ComputingReturn {
    TT_OK = 0,
    TT_TODO_ERROR = -1,
    TT_INPUT_ERROR = -2,
    TT_OUTPUT_ERROR = -3,
    TT_ATTR_ERROR = -4,
};

enum DataType {
    Float = 0,
    F16 = 1,
    BF16 = 2,
};

inline size_t DataType_size(DataType type_) {
    switch( type_ ) {
        case Float:
            return 4;
        case F16:
        case BF16:
            return 2;
        default:
            break;
    }
    tt_panic("Can't be here");
    return 0;
}

inline const char* DataType_name(DataType type_) {
    switch( type_ ) {
        case Float:
            return "f32";
        case F16:
            return "f16";
        case BF16:
            return "bf16";
        default:
            break;
    }
    tt_panic("Can't be here");
    return NULL;
}

// Logical/Math shape of a tensor
struct ShapeType {
public:
    ShapeType() {numel_ = 0;}
    ShapeType(const std::vector<size_t>& dims) {
        size_t ND = dims.size();
        dims_.resize(ND);
        for(size_t i = 0; i < ND; i++) {
            dims_[i] = dims[i];
        }
        numel_ = 0;
    }
    // all kinds accessors
    size_t numel() {
        if ( numel_ != 0) {
            return numel_;
        }
        numel_ = 1;
        for(size_t i = 0; i < dims_.size(); i++) {
            numel_ *= dims_[i];
        }
        return numel_;
    }
    const std::vector<size_t>& vec() const {
        return dims_;
    }
    const size_t* dims() const {
        return &dims_[0];
    }
    const size_t dim() const {
        return dims_.size();
    }
    bool operator == (const ShapeType& other) const {
        if ( other.dim() != dim() ) {
            return false;
        }
        for (size_t i = 0; i < dim(); i++) {
            if ( other.vec()[i] != dims_[i] ) {
                return false;
            }
        }
        return true;
    }
    bool operator != (const ShapeType& other) const {
        if ( other.dim() != dim() ) {
            return true;
        }
        for (size_t i = 0; i < dim(); i++) {
            if ( other.vec()[i] != dims_[i] ) {
                return true;
            }
        }
        return false;
    }
    std::string to_string() const {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < dim(); i++) {
            ss << dims_[i] << " ";
        }
        ss << "]";
        return ss.str();
    }

    std::vector<size_t> dense_strides() const {
        uint64_t s = 1;
        std::vector<size_t> stride;

        for (int i = dims_.size() - 1; i >= 0; i--) {
            stride.push_back(s);
            s = s * dims_[i];
        }
        std::reverse(stride.begin(), stride.end());
        return stride;
    }

private:
    std::vector<size_t>  dims_;
    size_t numel_;
};

// forward declare
template <DataType _DTYPE_> struct CPUTensor;
template <DataType _DTYPE_> struct CUDATensor;
using cpu_float_t = CPUTensor<DataType::Float>;
using cpu_bf16_t = CPUTensor<DataType::BF16>;
using cuda_float_t = CUDATensor<DataType::Float>;
using cuda_bf16_t = CUDATensor<DataType::BF16>;

struct TensorType;
using tensor_t = std::shared_ptr<tt::TensorType>;

// low level API for implementing Transformer
struct TransformerComputing {
    virtual ComputingReturn op_zero() {
        return TT_TODO_ERROR;
    }

    virtual ComputingReturn op_linear(tensor_t w, tensor_t bias, tensor_t y) {
        return TT_TODO_ERROR;
    }
    virtual ComputingReturn op_linear_backward(tensor_t x, tensor_t w, tensor_t _x, tensor_t _w, tensor_t _b) {
        return TT_TODO_ERROR;
    }

    virtual ComputingReturn op_transpos_0213(tensor_t y) {
        return TT_TODO_ERROR;
    }

    virtual ComputingReturn op_qk(tensor_t k, tensor_t qk) {
        return TT_TODO_ERROR;
    }
    virtual ComputingReturn op_qk_backward(tensor_t q, tensor_t k, tensor_t _q, tensor_t _k) {
        return TT_TODO_ERROR;
    }

    virtual ComputingReturn op_attn(tensor_t v, tensor_t attn) {
        return TT_TODO_ERROR;
    }
    virtual ComputingReturn op_attn_backward(tensor_t qk, tensor_t v, tensor_t _qk, tensor_t _v) {
        return TT_TODO_ERROR;
    }
};

// TensorType is all you need
struct TensorType: public TransformerComputing {
public:
    // init functions
    TensorType() = delete;
    TensorType(cpu_float_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Float), impl_(tensor) {};
    TensorType(cuda_float_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Float), impl_(tensor) {};
    TensorType(cpu_bf16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::BF16), impl_(tensor) {};
    TensorType(cuda_bf16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::BF16), impl_(tensor) {};
    ~TensorType();

    // fast access
    const ShapeType& shape() const {
        return shape_;
    }
    const DataType& dtype() const {
        return dtype_;
    }
    const size_t items() {
        return shape_.numel();
    }
    size_t impl_index() const {
        return impl_.index();
    }
    cpu_float_t* cpu_float() {
        if ( impl_.index() != CPU_FLOAT ) {
            tt_panic("Cant get cpu_float from a tensor");
        }
        return std::get<CPU_FLOAT>(impl_);
    }
    cpu_bf16_t* cpu_bf16() {
        if ( impl_.index() != CPU_BF16 ) {
            tt_panic("Cant get cpu_bf16 from a tensor");
        }
        return std::get<CPU_BF16>(impl_);
    }
    cuda_float_t* cuda_float() {
        if ( impl_.index() != CUDA_FLOAT ) {
            tt_panic("Cant get cuda_float from a tensor");
        }
        return std::get<CUDA_FLOAT>(impl_);
    }
    cuda_bf16_t* cuda_bf16() {
        if ( impl_.index() != CUDA_BF16 ) {
            tt_panic("Cant get cuda_bf16 from a tensor");
        }
        return std::get<CUDA_BF16>(impl_);
    }

    // help functions
    std::string to_string() {
        std::stringstream ss;
        ss << device_name() << ":" <<  DataType_name( dtype() );
        ss << ":[";
        for (size_t i = 0; i < shape_.vec().size(); i++) {
            ss << shape_.vec()[i];
            if (i != shape_.dim() - 1) {
                ss << " ";
            }
        }
        ss << "]";
        return ss.str();
    }
    const char* device_name() {
        if (impl_index() == ImplType::CPU_FLOAT) {
            return "cpu";
        }
        if (impl_index() == ImplType::CPU_BF16) {
            return "cpu";
        }
        return "cuda";
    }

    bool is_cpu() const {
        if (impl_index() == ImplType::CPU_FLOAT) {
            return true;
        }
        if (impl_index() == ImplType::CPU_BF16) {
            return true;
        }
        return false;
    }

    bool is_cuda() const {
        if (impl_index() == ImplType::CUDA_FLOAT) {
            return true;
        }
        if (impl_index() == ImplType::CUDA_BF16) {
            return true;
        }
        return false;
    }

    bool same_impl(tensor_t& other) {
        if ( impl_index() != other->impl_index() ) {
            return false;
        }
        return true;
    }
    bool same_dtype(tensor_t& other) {
        if ( dtype_ != other->dtype() ) {
            return false;
        }
        return true;
    }
    bool same_shape(tensor_t& other) {
        if ( shape_ != other->shape() ) {
            return false;
        }
        return true;
    }

    TransformerComputing* impl();

public:
    virtual ComputingReturn op_zero() {
        auto ret = impl()->op_zero();
        tt_check(ret, "zero");
    }
    virtual ComputingReturn op_linear(tensor_t w, tensor_t b, tensor_t y) {
        auto ret = impl()->op_linear(w, b, y);
        tt_check(ret, "linear");
    }
    virtual ComputingReturn op_linear_backward(tensor_t x, tensor_t w, tensor_t _x, tensor_t _w, tensor_t _b) {
        auto ret = impl()->op_linear_backward(x, w, _x, _w, _b);
        tt_check(ret, "linear_backward");
    }
    virtual ComputingReturn op_transpos_0213(tensor_t y) {
        auto ret = impl()->op_transpos_0213(y);
        tt_check(ret, "transpose_0213");
    }
    virtual ComputingReturn op_qk(tensor_t k, tensor_t qk) {
        auto ret = impl()->op_qk(k, qk);
        tt_check(ret, "qk");
    }
    virtual ComputingReturn op_qk_backward(tensor_t q, tensor_t k, tensor_t _q, tensor_t _k) {
        auto ret = impl()->op_qk_backward(q, k, _q, _k);
        tt_check(ret, "qk_backward");
    }
    virtual ComputingReturn op_attn(tensor_t v, tensor_t attn) {
        auto ret = impl()->op_attn(v, attn);
        tt_check(ret, "attn");
    }
    virtual ComputingReturn op_attn_backward(tensor_t qk, tensor_t v, tensor_t _qk, tensor_t _v) {
        auto ret = impl()->op_attn_backward(qk, v, _qk, _v);
        tt_check(ret, "attn_backward");
    }

    /*
    virtual std::variant<ComputingReturn, std::vector<tensor_t>> op_split_qkv(tensor_t x, int heads) {
        auto result = impl()->op_split_qkv(x, heads);
        if ( result.index() != 0) {
            ComputingReturn ret = std::get<0>(result);
            tt_check(ret, "split_qkv");
        }
        return result;
    }
    */

private:
    // basic info about tensor
    ShapeType shape_;
    const DataType  dtype_;

    // ImplType enum order is same as TensorImpl's variant
    enum ImplType {
        CPU_FLOAT,
        CPU_BF16,
        CUDA_FLOAT,
        CUDA_BF16,
    };
    using TensorImpl =   std::variant<  cpu_float_t*,
                                        cpu_bf16_t*,
                                        cuda_float_t*,
                                        cuda_bf16_t* >;

    TensorImpl impl_;
};

// Public interfaces without impl
struct ComputingContext {
    static int cuda_device;
    static cudaStream_t cuda_stream;
    static cublasHandle_t cublas_handle;
    static cublasLtHandle_t cublasLt_handle;
    static cudnnHandle_t cudnn_handle;

    static void* cuda_workspace;
    static size_t cuda_workspace_size;

    static void boot(int cud);
    static void shutdown();
};

tensor_t create_cuda_float(std::vector<size_t> shape_);
tensor_t create_cuda_bf16(std::vector<size_t> shape_);

} // end of namespace tt

#endif
