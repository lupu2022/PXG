#include "tensortype.hpp"
#include "cpu_impl.hpp"
#include "cuda_impl.hpp"

namespace tt {

TensorType::~TensorType() {

}

TransformerComputing* TensorType::impl() {
    return nullptr;
}

}

