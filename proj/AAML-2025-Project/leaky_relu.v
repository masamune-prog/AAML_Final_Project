module leaky_relu(
  input               clk,             // Added Clock
  input               reset,           // Added Reset
  input signed [31:0] input_data,
  input signed [31:0] input_offset,
  input signed [31:0] output_offset,
  input signed [31:0] multiplier_id,
  input signed [31:0] multiplier_alpla,
  input signed [31:0] shift_id,
  input signed [31:0] shift_alpla,
  input signed [31:0] min,
  input signed [31:0] max,
  output reg signed [31:0] clamped_output
);

  // --- Pipeline Stage 1: Preparation ---
  // Operation: Subtraction, Parameter Muxing, and Dynamic Shifting
  // This breaks the first part of the long chain.
  reg signed [31:0] s1_val_shift;
  reg signed [31:0] s1_multiplier;
  reg signed [31:0] s1_output_offset, s1_min, s1_max;
  reg signed [5:0]  s1_right_shift;

  always @(posedge clk) begin
    if (reset) begin
        s1_val_shift <= 0;
        s1_multiplier <= 0;
        s1_right_shift <= 0;
        s1_output_offset <= 0;
        s1_min <= 0;
        s1_max <= 0;
    end else begin
        // Combinational logic moved inside register logic
        reg signed [31:0] val;
        reg signed [5:0] raw_shift;
        
        val = input_data - input_offset;
        
        // Muxing logic to select Alpha or ID params
        if(val[31]) begin
             s1_multiplier <= multiplier_alpla;
             raw_shift = shift_alpla[5:0];
        end else begin
             s1_multiplier <= multiplier_id;
             raw_shift = shift_id[5:0];
        end

        // Shifting logic
        if($signed(raw_shift) > 0) begin
             s1_val_shift <= val <<< raw_shift;
             s1_right_shift <= 0;
        end else begin
             s1_val_shift <= val;
             s1_right_shift <= -$signed(raw_shift);
        end
        
        // Pass constants to next stage
        s1_output_offset <= output_offset;
        s1_min <= min;
        s1_max <= max;
    end
  end

  // --- Pipeline Stage 2: Multiplication & Accumulation ---
  // Operation: 32x32 Multiplication + Nudge + High bit extraction
  // By isolating the multiplier, Vivado can place this in DSP48 slices efficiently.
  reg signed [31:0] s2_high_mul_result;
  reg signed [5:0]  s2_right_shift;
  reg signed [31:0] s2_output_offset, s2_min, s2_max;

  always @(posedge clk) begin
    if (reset) begin
        s2_high_mul_result <= 0;
        s2_right_shift <= 0; 
    end else begin
      reg signed [63:0] product;
      reg signed [63:0] nudge;
      reg signed [63:0] acc;
      
      // The heavy math
      product = s1_val_shift * s1_multiplier;
      nudge = (product >= 0) ? (1 << 30) : (1 - (1 << 30));
      acc = product + nudge;

      // Logic for high result extraction
      if(s1_val_shift == -2147483648 && s1_multiplier == -2147483648) begin
           s2_high_mul_result <= 2147483647;
      end else begin
           s2_high_mul_result <= (acc < 0 && acc[30:0] != 0) ? acc[62:31] + 1 : acc[62:31];
      end

      // Pass through
      s2_right_shift <= s1_right_shift;
      s2_output_offset <= s1_output_offset;
      s2_min <= s1_min;
      s2_max <= s1_max;
    end
  end

  // --- Pipeline Stage 3: Scaling & Clamping ---
  // Operation: Rounding, Offset Addition, Min/Max Clamping
  always @(posedge clk) begin
    if (reset) begin
        clamped_output <= 0;
    end else begin
      reg [31:0] mask;
      reg [31:0] remainder;
      reg [31:0] threshold;
      reg signed [31:0] final_scaled_val;
      reg signed [31:0] unclamped;

      mask = (1 << s2_right_shift) - 1;
      remainder = s2_high_mul_result & mask;
      threshold = (mask >> 1) + (s2_high_mul_result < 0 ? 1 : 0);
      
      final_scaled_val = (s2_high_mul_result >>> s2_right_shift) + (remainder > threshold ? 1 : 0);
      unclamped = s2_output_offset + final_scaled_val;

      // Clamping
      if (unclamped > s2_max) begin
          clamped_output <= s2_max;
      end else if (unclamped < s2_min) begin
          clamped_output <= s2_min;
      end else begin
          clamped_output <= unclamped;
      end
    end
  end

endmodule