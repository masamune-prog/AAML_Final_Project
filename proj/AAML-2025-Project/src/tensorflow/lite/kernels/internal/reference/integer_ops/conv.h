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
// static uint32_t im2col_packed[512][8192];
// static uint32_t fr2row_packed[8192][512];

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

  // @ Test if this enhances the stability
  // memset(im2col_packed, 0, sizeof(im2col_packed));
  // memset(fr2row_packed, 0, sizeof(fr2row_packed));
  uint32_t im2col_packed[512][8192];
  uint32_t fr2row_packed[8192][512];

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

  // ==============================================================
  // STEP 3: HIGH-SPEED TILED EXECUTION (Safe 16x16)
  // ==============================================================

  cfu_op0(1, 0, 0);              // Reset
  cfu_op0(18, input_offset, 0);  // Input Offset

  const int K = filter_height * filter_width * input_depth;
  
  // 1. SAFETY CHECK: CALCULATE MAX TILE SIZE
  // Hardware buffer depth (entries per buffer A/B)
  const int BUFFER_DEPTH = 32768; 
  
  // How many "strips" (groups of 4) can we fit?
  // 1 strip = 1*K entries. 
  int max_strips = BUFFER_DEPTH / K;
  
  // Clamp strips to valid tile configs: 4 (16x16), 2 (8x8), or 1 (4x4)
  if (max_strips >= 8) max_strips = 8;
  else if (max_strips >= 4) max_strips = 4;
  else if (max_strips >= 2) max_strips = 2;
  else max_strips = 1;

  // Set the Tile Size based on safety calc
  // strips=4 -> Size 16. strips=2 -> Size 8. strips=1 -> Size 4.
  const int TILE_SIZE = max_strips * 4;

  cfu_op0(4, TILE_SIZE, 0); // Set M
  cfu_op0(6, TILE_SIZE, 0); // Set N

  for (int out_channel = 0; out_channel < output_depth; out_channel += TILE_SIZE) {
    
    int current_N = std::min(TILE_SIZE, output_depth - out_channel);
    cfu_op0(6, current_N, 0); 

    int weight_col_base = out_channel >> 2; 

    for (int slide = 0; slide < output_height * output_width; slide += TILE_SIZE) {
      
      int current_M = std::min(TILE_SIZE, (output_height * output_width) - slide);
      cfu_op0(4, current_M, 0); 

      int input_row_base = slide >> 2;

      // -------------------------------------------------------------
      // LOAD BUFFERS (Using op 19 with Safe Strip Count)
      // -------------------------------------------------------------
      cfu_op0(2, K, 0);  
      cfu_op0(20, 0, 0); // Reset K counter

      // We load 'max_strips' to ensure we never overflow the buffer.
      // Even if TILE_SIZE=16, if K is huge, max_strips might be 1 (Tile=4).
      for (int strip = 0; strip < max_strips; ++strip) {
        
        int packed_row = input_row_base + strip;
        int packed_col = weight_col_base + strip;

        // Check if we are at edge of image/channel limits
        // (Use (current_X + 3)/4 to get the number of needed strips)
        bool valid_row = (strip < (current_M + 3) / 4);
        bool valid_col = (strip < (current_N + 3) / 4);
        
        if (valid_row && valid_col) {
            for (int k = 0; k < K; ++k) {
                cfu_op0(19, 
                    im2col_packed[packed_row][k], 
                    fr2row_packed[k][packed_col]
                );
            }
        } else {
            // Padding: Write 0s if we are past the edge of the image
            for (int k = 0; k < K; ++k) {
                uint32_t val_A = valid_row ? im2col_packed[packed_row][k] : 0;
                uint32_t val_B = valid_col ? fr2row_packed[k][packed_col] : 0;
                cfu_op0(19, val_A, val_B);
            }
        }
      }

      // -------------------------------------------------------------
      // COMPUTE
      // -------------------------------------------------------------
      cfu_op0(12, 0, 0); 
      while (cfu_op0(13, 0, 0) != 0) { } 

      // -------------------------------------------------------------
      // READ BACK max tile = 32
      // -------------------------------------------------------------
      // int32_t acc[16][16];
      // ! fix here
      int32_t acc[32][32];
      
      int c_sram_addr = 0;
      
      // Calculate how many blocks the hardware actually processed
      int strips_M_proc = (current_M + 3) / 4;
      int strips_N_proc = (current_N + 3) / 4;

      for (int n_block = 0; n_block < strips_N_proc; ++n_block) {
          for (int m_block = 0; m_block < strips_M_proc; ++m_block) {
              for (int r = 0; r < 4; ++r) {
                  int actual_row = (m_block * 4) + r;
                  int actual_col_base = (n_block * 4);
                  
                  // Only read valid rows to avoid buffer overrun
                  if (actual_row < current_M) {
                      acc[actual_row][actual_col_base + 0] = cfu_op0(17, c_sram_addr, 0);
                      acc[actual_row][actual_col_base + 1] = cfu_op0(16, c_sram_addr, 0);
                      acc[actual_row][actual_col_base + 2] = cfu_op0(15, c_sram_addr, 0);
                      acc[actual_row][actual_col_base + 3] = cfu_op0(14, c_sram_addr, 0);
                  }
                  c_sram_addr++;
              }
          }
      }

      // -------------------------------------------------------------
      // POST PROCESSING
      // -------------------------------------------------------------
      for (int s = 0; s < current_M; ++s) {
        int current_slide = slide + s;
        if (current_slide >= output_height * output_width) continue;

        for (int c = 0; c < current_N; ++c) {
          int final_channel = out_channel + c;
          if (final_channel >= output_depth) continue;

          int32_t final_val = acc[s][c];
          
          if (bias_data) final_val += bias_data[final_channel];

          final_val = MultiplyByQuantizedMultiplier(
              final_val, output_multiplier[final_channel],
              output_shift[final_channel]);
          
          final_val += output_offset;
          final_val = std::max(final_val, output_activation_min);
          final_val = std::min(final_val, output_activation_max);

          int out_y = current_slide / output_width;
          int out_x = current_slide % output_width;
          
          output_data[Offset(output_shape, 0, out_y, out_x, final_channel)] =
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

  // ==============================================================
  // FINISHED
  // ==============================================================

}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
