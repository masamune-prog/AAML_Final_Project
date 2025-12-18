/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>
#include <cstring>

#include "cfu.h"
#include "perf.h"
#include "playground_util/print_params.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
// #pragma GCC optimize("Ofast,inline")
// the compiler flag is slower?
static uint32_t im2col_packed[512][8192];
static uint32_t fr2row_packed[8192][512];

namespace tflite {
namespace reference_integer_ops {

inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  perf_enable_counter(6);

  // --- Parameter Setup ---
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  // ===============================================================
  // STEP 1: FILL & PACK IM2COL (INPUTS)
  // ===============================================================

  int row_idx = 0;
  int col_idx = 0;

  // REMOVED: uint32_t packed_val = 0;
  // REMOVED: int pack_counter = 0;

  for (int out_y = 0; out_y < output_height; ++out_y) {
    const int top_edge = (out_y * stride_height) - pad_height;
    for (int out_x = 0; out_x < output_width; ++out_x) {
      const int left_edge = (out_x * stride_width) - pad_width;

      // Optimization: Calculate row/sub-row once
      int packed_row = row_idx >> 2;        // row / 4
      int byte_pos = row_idx & 3;           // row % 4
      int shift_amt = 24 - (byte_pos * 8);  // 24, 16, 8, 0

      for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        const int in_y = top_edge + filter_y * dilation_height_factor;
        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
          const int in_x = left_edge + filter_x * dilation_width_factor;
          const bool is_point_inside = (in_x >= 0 && in_x < input_width &&
                                        in_y >= 0 && in_y < input_height);

          for (int in_c = 0; in_c < filter_input_depth; ++in_c) {
            col_idx = filter_height * filter_width * in_c +
                      filter_y * filter_width + filter_x;

            uint8_t val;
            if (is_point_inside) {
              val = *((int8_t*)(input_data +
                                Offset(input_shape, 0, in_y, in_x, in_c)));
            } else {
              val = (uint8_t)(-input_offset);
            }

            // Read-Modify-Write to pack bits
            // Initialize with 0 if this is the first byte (byte_pos == 0)
            if (byte_pos == 0) {
              im2col_packed[packed_row][col_idx] = (val << 24);
            } else {
              im2col_packed[packed_row][col_idx] |= (val << shift_amt);
            }
          }
        }
      }
      row_idx++;
    }
  }

  // ===============================================================
  // STEP 2: FILL & PACK FR2ROW (WEIGHTS)
  // ===============================================================
  // We rearrange loops to pack 4 output channels into one uint32.
  // fr2row_packed[k][out_channel / 4]

  for (int out_c = 0; out_c < output_depth; out_c += 4) {
    // Handle boundary if depth % 4 != 0 (assuming multiple of 4 for simplicity
    // based on your code)

    for (int fy = 0; fy < filter_height; ++fy) {
      for (int fx = 0; fx < filter_width; ++fx) {
        for (int ic = 0; ic < filter_input_depth; ++ic) {
          int k = filter_height * filter_width * ic + fy * filter_width + fx;

          // Pack 4 channels: c, c+1, c+2, c+3
          uint32_t val32 = 0;
          for (int sub = 0; sub < 4; ++sub) {
            if (out_c + sub < output_depth) {
              uint8_t v =
                  *((int8_t*)(filter_data +
                              Offset(filter_shape, out_c + sub, fy, fx, ic)));
              val32 |= (v << (24 - sub * 8));
            }
          }
          fr2row_packed[k][out_c >> 2] = val32;
        }
      }
    }
  }

  // ===============================================================
  // STEP 3: HIGH-SPEED TILED EXECUTION
  // ===============================================================

  cfu_op0(1, 0, 0);              // Reset
  cfu_op0(18, input_offset, 0);  // Input Offset

  const int K = filter_height * filter_width * input_depth;

  // MATCHING YOUR HARDWARE: 12-bit address = 4096 entries
  const int MAX_CFU_SIZE = 8192;

  cfu_op0(4, 4, 0);  // M=4
  cfu_op0(6, 4, 0);  // N=4

  for (int out_channel = 0; out_channel < output_depth; out_channel += 4) {
    // Optimization: Pre-calculate column index for weights
    int weight_col_idx = out_channel >> 2;

    for (int slide = 0; slide < output_height * output_width; slide += 4) {
      // Optimization: Pre-calculate row index for inputs
      int input_row_idx = slide >> 2;

      int32_t acc[4][4] = {{0}};

      // --- TILE LOOP ---
      for (int k_start = 0; k_start < K; k_start += MAX_CFU_SIZE) {
        int k_chunk_size = std::min(MAX_CFU_SIZE, K - k_start);
        // printf("OC=%d, SLIDE=%d, K_START=%d, K_SIZE=%d\n", out_channel,
        // slide,
        //        k_start, k_chunk_size);
        // Only verify size if needed (some CFUs auto-reset index)
        cfu_op0(2, k_chunk_size, 0);

        // -------------------------------------------------------------
        // SUPER FAST LOADING LOOP
        // -------------------------------------------------------------
        // No shifting, no masking, just raw moves.
        cfu_op0(20, 0, 0);
        for (int k = 0; k < k_chunk_size; ++k) {
          int real_k = k_start + k;

          // // Load Weights: 1 Read, 1 Write
          // cfu_op0(10, k, fr2row_packed[real_k][weight_col_idx]);

          // // Load Inputs: 1 Read, 1 Write
          // cfu_op0(8, k, im2col_packed[input_row_idx][real_k]);
          cfu_op0(19,
                  im2col_packed[input_row_idx][real_k],  // Goes to Buffer A
                  fr2row_packed[real_k][weight_col_idx]  // Goes to Buffer B
          );
        }

        // Compute
        cfu_op0(12, 0, 0);  // Start
        while (cfu_op0(13, 0, 0) != 0) {
        }  // Wait

        // Accumulate
        for (int s = 0; s < 4; ++s) {
          acc[s][0] += cfu_op0(17, s, 0);
          acc[s][1] += cfu_op0(16, s, 0);
          acc[s][2] += cfu_op0(15, s, 0);
          acc[s][3] += cfu_op0(14, s, 0);
        }
      }

      // --- Post Processing ---
      for (int s = 0; s < 4; ++s) {
        int current_slide = slide + s;
        if (current_slide >= output_height * output_width) continue;

        for (int c = 0; c < 4; ++c) {
          if (out_channel + c >= output_depth) continue;

          int32_t final_val = acc[s][c];
          if (bias_data) final_val += bias_data[out_channel + c];

          final_val = MultiplyByQuantizedMultiplier(
              final_val, output_multiplier[out_channel + c],
              output_shift[out_channel + c]);
          final_val += output_offset;
          final_val = std::max(final_val, output_activation_min);
          final_val = std::min(final_val, output_activation_max);

          int out_y = current_slide / output_width;
          int out_x = current_slide % output_width;
          output_data[Offset(output_shape, 0, out_y, out_x, out_channel + c)] =
              static_cast<int8_t>(final_val);
        }
      }
    }
  }
  perf_disable_counter(6);
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int top_edge = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int left_edge = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = top_edge + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = left_edge + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
