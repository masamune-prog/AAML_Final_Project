/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "cfu.h"
namespace tflite {
namespace reference_ops {

inline void LeakyRelu(const tflite::LeakyReluParams& params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    // Note that alpha might be > 1 or < 0, so we don't use std::max here.
    output_data[i] = val > 0 ? val : val * params.alpha;
  }
}

template <typename T>
inline void QuantizeLeakyRelu(const LeakyReluParams& params,
                              const RuntimeShape& input_shape,
                              const T* input_data,
                              const RuntimeShape& output_shape,
                              T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static const int32_t quantized_min = std::numeric_limits<T>::min();
  static const int32_t quantized_max = std::numeric_limits<T>::max();

  /*
void ProcessData(const int8_t* input_data, int8_t* output_data, int flat_size, 
                 int32_t input_offset, int32_t output_offset,
                 int32_t mult_id, int shift_id, 
                 int32_t mult_alpha, int shift_alpha,
                 int32_t min_val, int32_t max_val) {

    for (int i = 0; i < flat_size; ++i) {
        // Step 1: Apply input offset
        int32_t x = (int32_t)input_data[i] - input_offset;

        // Step 2: Select the correct multiplier/shift based on if x is positive or negative
        int32_t multiplier = (x >= 0) ? mult_id : mult_alpha;
        int shift = (x >= 0) ? shift_id : shift_alpha;

        // Step 3: MultiplyByQuantizedMultiplier Logic (Fused)
        // Part A: Left Shift
        if (shift > 0) {
            x = x << shift;
        }
        int right_shift = (shift > 0) ? 0 : -shift;

        // Part B: Saturating Rounding Doubling High Mul
        int32_t high_mul_result;
        if (x == -2147483648 && multiplier == -2147483648) {
            high_mul_result = 2147483647; // Max int32
        } else {
            int64_t product = (int64_t)x * (int64_t)multiplier;
            // Rounding nudge: adds a bit before we divide by 2^31
            int64_t nudge = (product >= 0) ? (1LL << 30) : (1LL - (1LL << 30));
            high_mul_result = (int32_t)((product + nudge) / (1LL << 31));
        }

        // Part C: Rounding Divide By POT (Power of Two)
        int32_t final_scaled_val;
        if (right_shift > 0) {
            int32_t mask = (1 << right_shift) - 1;
            int32_t remainder = high_mul_result & mask;
            int32_t threshold = (mask >> 1) + (high_mul_result < 0 ? 1 : 0);
            final_scaled_val = (high_mul_result >> right_shift) + (remainder > threshold ? 1 : 0);
        } else {
            final_scaled_val = high_mul_result;
        }

        // Step 4: Add output offset and Clamp (Min/Max)
        int32_t unclamped = output_offset + final_scaled_val;
        
        if (unclamped < min_val) unclamped = min_val;
        if (unclamped > max_val) unclamped = max_val;

        output_data[i] = (int8_t)unclamped;
    }
}
  */
 //printf("quantized_min = %ld",quantized_min);
 //printf("quantized_max = %ld\n",quantized_max);
  cfu_op0(10, params.input_offset ,params.output_offset);
  cfu_op0(11, params.output_multiplier_identity ,params.output_multiplier_alpha);
  cfu_op0(12, params.output_shift_identity ,params.output_shift_alpha);
  cfu_op0(15, quantized_min ,quantized_max);
  for (int i = 0; i < flat_size; ++i) {
    const int32_t input_value_relu = input_data[i];
    cfu_op0(13, input_value_relu ,0);
    int32_t clamped_output = cfu_op0(14, 0, 0);
  /*
    const int32_t input_value = input_data[i] - params.input_offset;
    
    int32_t unclamped_output;
    if (input_value >= 0) {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier(
                             input_value, params.output_multiplier_identity,
                             params.output_shift_identity);
    } else {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier(
                             input_value, params.output_multiplier_alpha,
                             params.output_shift_alpha);
    }
    const T clamped_output_golden =
        std::min(quantized_max, std::max(quantized_min, unclamped_output));
    */
    output_data[i] = static_cast<T>(clamped_output);
  }
}

}  // namespace reference_ops
}  // namespace tflite


#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
