#include <cooperative_groups.h>
#include <cstddef>

#include "common.h"
#include "block_reduce.h"
#include "kernels.h"

namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32

template <typename T>
__forceinline__ __device__ T add_eps(T x) {
  return fabsf(x) > LN_EPSILON ? x : (x < 0 ? -LN_EPSILON : LN_EPSILON);
}

/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size* seq_len, hidden_size], ln result.
vars: [batch_size* seq_len], variance per token
means: [batch_size* seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  // step 0. compute local sum
  float l_sum = 0;
  float l_square_sum = 0;
  const float4 *inp_f4 =
      reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_square_sum +=
        val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // step 1. compute reduce sum
  float mean_dim = float(hidden_size) * 4.f;
  float reduce_val[2] = {l_sum, l_square_sum};
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float s_mean, s_var;
  if (threadIdx.x == 0) {
    s_mean = reduce_val[0] / mean_dim;
    if (means != nullptr) {
      means[blockIdx.x] = s_mean;
    }
    s_var = reduce_val[1] / mean_dim - s_mean * s_mean + LN_EPSILON;
    vars[blockIdx.x] = s_var;
    s_var = rsqrtf(s_var);
  }
  __syncthreads();

  // step 2. layer norm result
  float4 *output_f4 =
      reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 vscale = __ldg(reinterpret_cast<const float4 *>(scale) + idx);
    float4 vbias = __ldg(reinterpret_cast<const float4 *>(bias) + idx);
    float4 val = inp_f4[idx];
    val.x = (val.x - s_mean) * s_var * vscale.x + vbias.x;
    val.y = (val.y - s_mean) * s_var * vscale.y + vbias.y;
    val.z = (val.z - s_mean) * s_var * vscale.z + vbias.z;
    val.w = (val.w - s_mean) * s_var * vscale.w + vbias.w;
    output_f4[idx] = val;
  }
}

template <>
__global__ void ker_layer_norm<__half>(__half *ln_res, __half *vars,
                                       __half *means, const __half *inp,
                                       const __half *scale, const __half *bias,
                                       int hidden_size) {
  // step 0. compute local sum
  float l_sum = 0;
  float l_square_sum = 0;
  const float4 *inp_f4 =
      reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val_f4 = inp_f4[idx];
    __half2 *val_h2 = (__half2 *)(&val_f4);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 val_f2 = __half22float2(val_h2[i]);
      l_sum += val_f2.x + val_f2.y;
      l_square_sum += val_f2.x * val_f2.x + val_f2.y * val_f2.y;
    }
  }

  // step 1. compute reduce sum
  float mean_dim = float(hidden_size) * 8.f;
  float reduce_val[2] = {l_sum, l_square_sum};
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float s_mean, s_var;
  if (threadIdx.x == 0) {
    s_mean = reduce_val[0] / mean_dim;
    if (means != nullptr) {
      means[blockIdx.x] = s_mean;
    }
    s_var = reduce_val[1] / mean_dim - s_mean * s_mean + LN_EPSILON;
    vars[blockIdx.x] = s_var;
    s_var = rsqrtf(s_var);
  }
  __syncthreads();

  // step 2. layer norm result
  float4 *output_f4 =
      reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    // load scale, bias, input
    float4 scale_f4 = __ldg(reinterpret_cast<const float4 *>(scale) + idx);
    __half2 *scale_h2 = reinterpret_cast<__half2 *>(&scale_f4);
    float4 bias_f4 = __ldg(reinterpret_cast<const float4 *>(bias) + idx);
    __half2 *bias_h2 = reinterpret_cast<__half2 *>(&bias_f4);
    float4 val_f4 = inp_f4[idx];
    __half2 *val_h2 = reinterpret_cast<__half2 *>(&val_f4);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 scale_f2 = __half22float2(scale_h2[i]);
      float2 bias_f2 = __half22float2(bias_h2[i]);
      float2 val_f2 = __half22float2(val_h2[i]);
      val_f2.x = (val_f2.x - s_mean) * s_var * scale_f2.x + bias_f2.x;
      val_f2.y = (val_f2.y - s_mean) * s_var * scale_f2.y + bias_f2.y;
      val_h2[i] = __float22half2_rn(val_f2);
    }
    output_f4[idx] = val_f4;
  }
}

/*
void launch_layer_norm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
      throw ::std::runtime_error("violate hidden_dim % 4 = 0");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      ln_res, vars, means, inp, scale, bias, hidden_dim);
}
*/

template <typename T>
void launch_layer_norm(T *ln_res, T *vars, T *means,
                          const T *inp, const T *scale,
                          const T *bias, int batch_size, int hidden_dim,
                          cudaStream_t stream) {
  int n = 32 / sizeof(T);
  if (hidden_dim % n != 0) {
      throw ::std::runtime_error("violate hidden_dim % 4(float) 8(__half) = 0");
  }
  hidden_dim /= n;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<T><<<grid_dim, block_dim, 0, stream>>>(
      ln_res, vars, means, inp, scale, bias, hidden_dim);
}

template <>
void launch_layer_norm<float>(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream); 

template <>
void launch_layer_norm<__half>(__half *ln_res, __half *vars, __half *means,
                               const __half *inp, const __half *scale,
                               const __half *bias, int batch_size,
                               int hidden_dim, cudaStream_t stream);

}
