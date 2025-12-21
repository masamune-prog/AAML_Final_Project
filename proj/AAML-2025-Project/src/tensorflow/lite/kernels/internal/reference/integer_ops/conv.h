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

#include "playground_util/print_params.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
// #include <stdio.h>
#include "cfu.h"
#include "perf.h"

namespace tflite {

namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  perf_enable_counter(6);
  const int32_t input_offset = params.input_offset;

  int8_t default_val = static_cast<int8_t>(-input_offset);
  uint8_t default_u8 = static_cast<uint8_t>(default_val);
  uint32_t default_packed = (static_cast<uint32_t>(default_u8) << 24) |
                            (static_cast<uint32_t>(default_u8) << 16) |
                            (static_cast<uint32_t>(default_u8) << 8) |
                            static_cast<uint32_t>(default_u8);

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // const int batches = input_shape.Dims(0);
  const int output_depth = output_shape.Dims(3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int filter_H_MUL_filter_W = filter_height * filter_width;
  const int im2col_h = output_height * output_width;
  const int im2col_w = filter_H_MUL_filter_W * filter_input_depth;
  // const int kernel_h = im2col_w;
  const int kernel_w = output_depth;

  constexpr int MAX_IM2COL_H = 2048;
  constexpr int MAX_IM2COL_W = 8192;
  constexpr int MAX_KERNEL_H = 8192;
  constexpr int MAX_KERNEL_W = 2048;
  constexpr int MAX_RESULT_H = 2048;
  constexpr int MAX_RESULT_W = 2048;

  // printf("im2col_h = %d", im2col_h);
  // printf("im2col_w = %d", im2col_w);
  // printf("kernel_w = %d", kernel_w);
  // printf("filter_H_MUL_filter_W = %d\n", filter_H_MUL_filter_W);

  int8_t im2col[MAX_IM2COL_H * MAX_IM2COL_W];
  int8_t kernel[MAX_KERNEL_H * MAX_KERNEL_W];
  int32_t output_buf[MAX_RESULT_H * MAX_RESULT_W];

  constexpr int T = 32;
  constexpr int SYSTOLIC_SIZE = 4;
  uint8_t K_in = T, M_in = T, N_in = T;
  // for (int batch = 0; batch < batches; ++batch) {
  // build im2col
  for (int out_y = 0; out_y < output_height; ++out_y) {
    const int in_y_origin = out_y * stride_height - pad_height;
    const int out_y_offset = out_y * output_width;
    for (int out_x = 0; out_x < output_width; ++out_x) {
      const int in_x_origin = out_x * stride_width - pad_width;
      const int im2col_i = out_y_offset + out_x;
      const int im2col_row_base = im2col_i * im2col_w;

      for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        const int in_y = in_y_origin + dilation_height_factor * filter_y;
        const int filter_y_offset = filter_y * filter_width;

        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
          const int in_x = in_x_origin + dilation_width_factor * filter_x;
          const bool is_point_inside_image =
              (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
              (in_y < input_height);

          // Unrolling in_channel by 4
          for (int in_channel = 0; in_channel < filter_input_depth;
               in_channel += 4) {
            int base =
                in_channel * filter_H_MUL_filter_W + filter_y_offset + filter_x;

            int im2col_j0 = base;
            int im2col_j1 =
                base + filter_H_MUL_filter_W;  // 若 filter_H_MUL_filter_W =
                                               // filter_height*filter_width
            int im2col_j2 = base + 2 * filter_H_MUL_filter_W;
            int im2col_j3 = base + 3 * filter_H_MUL_filter_W;

            im2col[im2col_row_base + im2col_j0] =
                is_point_inside_image ? input_data[Offset(input_shape, 0, in_y,
                                                          in_x, in_channel + 0)]
                                      : static_cast<int8_t>(-input_offset);

            im2col[im2col_row_base + im2col_j1] =
                is_point_inside_image ? input_data[Offset(input_shape, 0, in_y,
                                                          in_x, in_channel + 1)]
                                      : static_cast<int8_t>(-input_offset);

            im2col[im2col_row_base + im2col_j2] =
                is_point_inside_image ? input_data[Offset(input_shape, 0, in_y,
                                                          in_x, in_channel + 2)]
                                      : static_cast<int8_t>(-input_offset);

            im2col[im2col_row_base + im2col_j3] =
                is_point_inside_image ? input_data[Offset(input_shape, 0, in_y,
                                                          in_x, in_channel + 3)]
                                      : static_cast<int8_t>(-input_offset);
          }
        }
      }
    }
  }

  // build kernel
  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
      const int filter_y_offset = filter_y * filter_width;
      for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
        const int filter_x_offset = filter_y_offset + filter_x;
        // Unrolling in_channel by 4
        for (int in_channel = 0; in_channel < filter_input_depth;
             in_channel += 4) {
          int base =
              filter_x_offset + in_channel * filter_height * filter_width;

          int kernel_i0 = base;
          int kernel_i1 = base + filter_height * filter_width;
          int kernel_i2 = base + 2 * filter_height * filter_width;
          int kernel_i3 = base + 3 * filter_height * filter_width;

          kernel[kernel_i0 * kernel_w + out_channel] = filter_data[Offset(
              filter_shape, out_channel, filter_y, filter_x, in_channel + 0)];

          kernel[kernel_i1 * kernel_w + out_channel] = filter_data[Offset(
              filter_shape, out_channel, filter_y, filter_x, in_channel + 1)];

          kernel[kernel_i2 * kernel_w + out_channel] = filter_data[Offset(
              filter_shape, out_channel, filter_y, filter_x, in_channel + 2)];

          kernel[kernel_i3 * kernel_w + out_channel] = filter_data[Offset(
              filter_shape, out_channel, filter_y, filter_x, in_channel + 3)];
        }
      }
    }
  }

  cfu_op0(6, input_offset, 0);  // Set input offset

  // Initialize output_buf (faster than using {0} to ini)
  for (int i = 0; i < im2col_h; ++i) {
    int row_base = i * kernel_w;
    for (int j = 0; j < kernel_w; ++j) {
      output_buf[row_base + j] = 0;
    }
  }

  // Tiled GEMM with CFU
  for (int m = 0; m < im2col_h; m += T) {
    int m_tile = std::min(m + T, im2col_h) - m;
    int m_quotient = m_tile / 4;
    int m_remainder = m_tile % 4;
    for (int n = 0; n < kernel_w; n += T) {
      int n_tile = std::min(n + T, kernel_w) - n;
      int n_quotient = n_tile / 4;
      int n_remainder = n_tile % 4;

      for (int k = 0; k < im2col_w; k += T) {
        int k_tile = std::min(k + T, im2col_w) - k;

        cfu_op0(0, 0, 0);  // Initialize systolic array

        // pre-process im2col data for memory mapping (Type A)
        // handle % 4 == 0
        for (int col_block = 0; col_block < m_quotient; ++col_block) {
          int col_start = col_block * SYSTOLIC_SIZE;
          for (int i = 0; i < k_tile; ++i) {
            // unroll 4 times and put im2col data into cfu
            uint8_t v0 = static_cast<uint8_t>(
                im2col[(m + col_start) * im2col_w + (k + i)]);
            uint8_t v1 = static_cast<uint8_t>(
                im2col[(m + col_start + 1) * im2col_w + (k + i)]);
            uint8_t v2 = static_cast<uint8_t>(
                im2col[(m + col_start + 2) * im2col_w + (k + i)]);
            uint8_t v3 = static_cast<uint8_t>(
                im2col[(m + col_start + 3) * im2col_w + (k + i)]);

            // packed 4 int8_t to 1 int32_t
            uint32_t A_packed = ((uint32_t)v0 << 24) | ((uint32_t)v1 << 16) |
                                ((uint32_t)v2 << 8) | (uint32_t)v3;
            cfu_op0(1, A_packed, 0);  // send 32 bit data into cfu
          }
        }

        // handle %4 != 0
        if (m_remainder > 0) {
          int col_start = m_quotient * SYSTOLIC_SIZE;
          for (int i = 0; i < k_tile; ++i) {
            uint32_t A_packed =
                default_packed;  // pre-filled -input_offset for edge cases

            // write value based on m_remainder
            if (m_remainder > 0) {
              uint8_t v0 = static_cast<uint8_t>(
                  im2col[(m + col_start) * im2col_w + (k + i)]);
              A_packed = (A_packed & 0x00FFFFFFu) | ((uint32_t)v0 << 24);
            }
            if (m_remainder > 1) {
              uint8_t v1 = static_cast<uint8_t>(
                  im2col[(m + col_start + 1) * im2col_w + (k + i)]);
              A_packed = (A_packed & 0xFF00FFFFu) | ((uint32_t)v1 << 16);
            }
            if (m_remainder > 2) {
              uint8_t v2 = static_cast<uint8_t>(
                  im2col[(m + col_start + 2) * im2col_w + (k + i)]);
              A_packed = (A_packed & 0xFFFF00FFu) | ((uint32_t)v2 << 8);
            }

            cfu_op0(1, A_packed, 0);
          }
        }
        // preprocess im2col ends here

        // preprocess kernel data (same logic as im2col but mem mapping is type
        // B)
        for (int col_block = 0; col_block < n_quotient; ++col_block) {
          int col_start = col_block * SYSTOLIC_SIZE;
          for (int i = 0; i < k_tile; ++i) {
            // unroll 4 times and put kernel data into cfu
            uint8_t v0 = static_cast<uint8_t>(
                kernel[(k + i) * kernel_w + (n + col_start)]);
            uint8_t v1 = static_cast<uint8_t>(
                kernel[(k + i) * kernel_w + (n + col_start + 1)]);
            uint8_t v2 = static_cast<uint8_t>(
                kernel[(k + i) * kernel_w + (n + col_start + 2)]);
            uint8_t v3 = static_cast<uint8_t>(
                kernel[(k + i) * kernel_w + (n + col_start + 3)]);

            // packed 4 int8_t to 1 int32_t
            uint32_t B_packed = ((uint32_t)v0 << 24) | ((uint32_t)v1 << 16) |
                                ((uint32_t)v2 << 8) | (uint32_t)v3;
            cfu_op0(2, 0, B_packed);  // send 32 bit data into cfu
          }
        }

        if (n_remainder > 0) {
          int col_start = n_quotient * SYSTOLIC_SIZE;
          for (int i = 0; i < k_tile; ++i) {
            uint32_t B_packed = 0;  // pre-filled 0 for edge cases

            if (n_remainder > 0) {
              uint8_t v0 = static_cast<uint8_t>(
                  kernel[(k + i) * kernel_w + (n + col_start)]);
              B_packed = (B_packed & 0x00FFFFFFu) | ((uint32_t)v0 << 24);
            }
            if (n_remainder > 1) {
              uint8_t v1 = static_cast<uint8_t>(
                  kernel[(k + i) * kernel_w + (n + col_start + 1)]);
              B_packed = (B_packed & 0xFF00FFFFu) | ((uint32_t)v1 << 16);
            }
            if (n_remainder > 2) {
              uint8_t v2 = static_cast<uint8_t>(
                  kernel[(k + i) * kernel_w + (n + col_start + 2)]);
              B_packed = (B_packed & 0xFFFF00FFu) | ((uint32_t)v2 << 8);
            }

            cfu_op0(2, 0, B_packed);  // send 32 bit data into cfu
          }
        }
        // preprocess kernel ends here

        M_in = m_tile;
        N_in = n_tile;
        K_in = k_tile;
        cfu_op0(3, ((K_in << 16) | (M_in << 8) | N_in), 0);  // Set dimensions
        while (cfu_op0(4, 0, 0)) {                           /* busy wait */
        }

        // handle output data from cfu
        // first handle % 4 == 0
        for (int current_n = 0; current_n < n_quotient; ++current_n) {
          int base_col =
              4 * current_n;  // calc base index to directly put into output_buf
          for (int i = 0; i < m_tile; ++i) {
            int out_row_base = (m + i) * kernel_w;
            // unroll 4 times directly put back to output_buf
            int32_t val0 = cfu_op0(5, 0, 0);
            int32_t val1 = cfu_op0(5, 0, 0);
            int32_t val2 = cfu_op0(5, 0, 0);
            int32_t val3 = cfu_op0(5, 0, 0);

            output_buf[out_row_base + (n + base_col)] += val0;
            output_buf[out_row_base + (n + base_col + 1)] += val1;
            output_buf[out_row_base + (n + base_col + 2)] += val2;
            output_buf[out_row_base + (n + base_col + 3)] += val3;
          }
        }

        // handle % 4 != 0
        if (n_remainder > 0) {
          int base_col = 4 * n_quotient;
          for (int i = 0; i < m_tile; ++i) {
            int out_row_base = (m + i) * kernel_w;

            int32_t val0 = cfu_op0(5, 0, 0);
            int32_t val1 = cfu_op0(5, 0, 0);
            int32_t val2 = cfu_op0(5, 0, 0);
            int32_t val3 = cfu_op0(5, 0, 0);

            if (n_remainder > 0)
              output_buf[out_row_base + (n + base_col)] += val0;
            if (n_remainder > 1)
              output_buf[out_row_base + (n + base_col + 1)] += val1;
            if (n_remainder > 2)
              output_buf[out_row_base + (n + base_col + 2)] += val2;
            (void)val3;  // let compiler knows val3 has been used
          }
        }
      }
    }
  }

  for (int out_y = 0; out_y < output_height; ++out_y) {
    const int base_y = (0 * output_height + out_y) * output_width;
    for (int out_x = 0; out_x < output_width; ++out_x) {
      const int base_x = base_y + out_x;  // calc base index for x
      const int out_row_base = (out_y * output_width + out_x) *
                               kernel_w;  // calc output row base for output_buf

      int output_index_base =
          base_x * output_depth;  // calc output index base for output data
      // use ptr to reduce calc for output
      int32_t* buf_ptr = &output_buf[out_row_base];
      int8_t* output_ptr = &output_data[output_index_base];

      for (int out_channel = 0; out_channel < output_depth; out_channel += 4) {
        int32_t acc0 = buf_ptr[out_channel];
        int32_t acc1 = buf_ptr[out_channel + 1];
        int32_t acc2 = buf_ptr[out_channel + 2];
        int32_t acc3 = buf_ptr[out_channel + 3];

        if (bias_data) {
          acc0 += bias_data[out_channel];
          acc1 += bias_data[out_channel + 1];
          acc2 += bias_data[out_channel + 2];
          acc3 += bias_data[out_channel + 3];
        }

        acc0 = MultiplyByQuantizedMultiplier(
            acc0, output_multiplier[out_channel], output_shift[out_channel]);
        acc1 = MultiplyByQuantizedMultiplier(acc1,
                                             output_multiplier[out_channel + 1],
                                             output_shift[out_channel + 1]);
        acc2 = MultiplyByQuantizedMultiplier(acc2,
                                             output_multiplier[out_channel + 2],
                                             output_shift[out_channel + 2]);
        acc3 = MultiplyByQuantizedMultiplier(acc3,
                                             output_multiplier[out_channel + 3],
                                             output_shift[out_channel + 3]);

        acc0 += output_offset;
        acc1 += output_offset;
        acc2 += output_offset;
        acc3 += output_offset;

        acc0 = std::max(acc0, output_activation_min);
        acc1 = std::max(acc1, output_activation_min);
        acc2 = std::max(acc2, output_activation_min);
        acc3 = std::max(acc3, output_activation_min);

        acc0 = std::min(acc0, output_activation_max);
        acc1 = std::min(acc1, output_activation_max);
        acc2 = std::min(acc2, output_activation_max);
        acc3 = std::min(acc3, output_activation_max);

        output_ptr[out_channel] = static_cast<int8_t>(acc0);
        output_ptr[out_channel + 1] = static_cast<int8_t>(acc1);
        output_ptr[out_channel + 2] = static_cast<int8_t>(acc2);
        output_ptr[out_channel + 3] = static_cast<int8_t>(acc3);
      }
    }
  }
  perf_enable_counter(6);
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
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

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