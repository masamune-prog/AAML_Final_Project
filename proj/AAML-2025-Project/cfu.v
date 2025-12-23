module Cfu (
  input               cmd_valid,
  output reg          cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

  reg             in_valid;
  reg [15:0]      K;
  reg [15:0]      M;
  reg [15:0]      N;
  wire            busy;
  wire A_wr_en;
  reg  A_wr_en_fromCFU;
  wire A_wr_en_fromTPU;
  wire B_wr_en;
  reg B_wr_en_fromCFU;
  wire B_wr_en_fromTPU;
  wire C_wr_en;
  reg C_wr_en_fromCFU;
  wire C_wr_en_fromTPU;
  reg print;
  wire [15:0] A_index;
  wire [15:0] A_index_forRead;
  reg [15:0] A_index_forPrint;
  reg [15:0] A_index_forWrite;
  wire [15:0] B_index;
  wire [15:0] B_index_forRead;
  wire [15:0] B_index_forPrint;
  reg [15:0] B_index_forWrite;
  wire [8:0] C_index;
  reg [8:0] C_index_forRead;
  wire [8:0] C_index_forWrite;
  reg [31:0] A_data_in;
  wire [31:0] A_data_out;
  reg [31:0] B_data_in;
  wire [31:0] B_data_out;
  wire [127:0] C_data_in;
  wire [127:0] C_data_out;
  reg [31:0] offset;
  reg [2:0] state_forOutput;
  wire [2:0] cur_state_fromTPU; 
  reg [1:0] readC_counter;

  // -- ReLU Registers --
  reg signed [31:0] relu_input_offset;
  reg signed [31:0] relu_output_offset;
  reg signed [31:0] relu_multiplier_id;
  reg signed [31:0] relu_multiplier_alpla;
  reg signed [31:0] relu_shift_id;
  reg signed [31:0] relu_shift_alpla;
  reg signed [31:0] relu_input;
  reg signed [31:0] relu_min;
  reg signed [31:0] relu_max;
  wire signed [31:0] relu_clamped_output;

  // -- State Machine --
  localparam READ_CFU  = 3'd0;
  localparam WAIT_RSP  = 3'd1;
  localparam WAIT_RELU = 3'd2; // New state for pipeline wait
  reg [2:0] cfu_state;
  reg [1:0] relu_latency_cnt;  // Counter for pipeline delay

  wire check_CFU_or_TPU;
  assign check_CFU_or_TPU = busy;
  
  assign A_index = check_CFU_or_TPU ? A_index_forRead : (print ? A_index_forPrint: A_index_forWrite);
  assign A_wr_en = check_CFU_or_TPU ? 0 : A_wr_en_fromCFU;
  assign B_index = check_CFU_or_TPU ? B_index_forRead : (print ? B_index_forPrint: B_index_forWrite);
  assign B_wr_en = check_CFU_or_TPU ? 0 : B_wr_en_fromCFU;
  assign C_index = check_CFU_or_TPU ? C_index_forWrite : C_index_forRead;
  assign C_wr_en = check_CFU_or_TPU ? C_wr_en_fromTPU : C_wr_en_fromCFU;

  global_buffer_bram #(
    .ADDR_BITS(16), 
    .DATA_BITS(32) 
  )
  gbuff_A(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(A_wr_en),
    .index(A_index),
    .data_in(A_data_in),
    .data_out(A_data_out) 
  );

  global_buffer_bram #(
    .ADDR_BITS(16), 
    .DATA_BITS(32) 
  )
  gbuff_B(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(B_wr_en),
    .index(B_index),
    .data_in(B_data_in),
    .data_out(B_data_out) 
  );
  
  global_buffer_bram #(
    .ADDR_BITS(9), 
    .DATA_BITS(128) 
  )
  gbuff_C(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(C_wr_en),
    .index(C_index),
    .data_in(C_data_in),
    .data_out(C_data_out) 
  );
  
  TPU My_TPU(
    .clk            (clk),  
    .reset          (reset),       
    .in_valid       (in_valid),         
    .K              (K), 
    .M              (M), 
    .N              (N), 
    .busy           (busy),        
    .A_wr_en        (),       
    .A_index        (A_index_forRead),      
    .A_data_in      (),       
    .A_data_out     (A_data_out),         
    .B_wr_en        (),       
    .B_index        (B_index_forRead),         
    .B_data_in      (),      
    .B_data_out     (B_data_out),         
    .C_wr_en        (C_wr_en_fromTPU),      
    .C_index        (C_index_forWrite),      
    .C_data_in      (C_data_in),      
    .C_data_out     (C_data_out),
    .offset         (offset)
  );

  // -- PATCHED: Pipelined ReLU --
  leaky_relu My_relu(
    .clk              (clk),           // Connected
    .reset            (reset),         // Connected
    .input_data       (relu_input),
    .input_offset     (relu_input_offset),
    .output_offset    (relu_output_offset),
    .multiplier_id    (relu_multiplier_id),
    .multiplier_alpla (relu_multiplier_alpla),
    .shift_id         (relu_shift_id),
    .shift_alpla      (relu_shift_alpla),
    .min              (relu_min),
    .max              (relu_max),
    .clamped_output   (relu_clamped_output)
  );

  always @(posedge clk) begin
    if (in_valid) begin
      in_valid <= 0;
    end
    if (reset) begin
      in_valid <= 1'b0;
      cmd_ready <= 1'b1;
      rsp_valid <= 1'b0;
      A_index_forWrite <= 12'b0;
      B_index_forWrite <= 12'd0;
      A_wr_en_fromCFU <= 1'b0;
      B_wr_en_fromCFU <= 1'b0;
      readC_counter <= 2'd0;
      state_forOutput <= 0;
      offset <= 0;
      cfu_state <= READ_CFU;
      print <= 0;
      relu_latency_cnt <= 0;
    end else begin
      case (cfu_state)
        READ_CFU: begin
          if (cmd_valid && cmd_ready) begin
            cmd_ready <= 1'b0;
            // Default response Valid, unless overridden below
            rsp_valid <= 1'b1; 
            cfu_state <= WAIT_RSP;

            case(cmd_payload_function_id[9:3])
              7'd0:begin
                A_wr_en_fromCFU <= 1'b0;
                B_wr_en_fromCFU <= 1'b0;
                A_index_forWrite <= 12'd0 - 12'd1; 
                B_index_forWrite <= 12'd0 - 12'd1; 
                print = 0;
              end
              7'd1:begin
                A_wr_en_fromCFU = 1'b1;
                A_index_forWrite <= A_index_forWrite + 1; 
                A_data_in <= cmd_payload_inputs_0[31:0];
              end
              7'd2:begin
                B_wr_en_fromCFU = 1'b1;
                B_index_forWrite <= B_index_forWrite + 1;
                B_data_in <= cmd_payload_inputs_1[31:0];
              end
              7'd3:begin
                in_valid <= 1'b1;
                K <= cmd_payload_inputs_0[23:16];
                M <= cmd_payload_inputs_0[15:8];
                N <= cmd_payload_inputs_0[7:0];
              end
              7'd4:begin
                if (busy) begin
                  rsp_payload_outputs_0 <= 1;
                end else begin 
                  rsp_payload_outputs_0 <= 0;
                  C_index_forRead <= 0;
                  readC_counter <= 2'd0;
                end
              end
              7'd5:begin
                C_wr_en_fromCFU = 1'b0;
                if (readC_counter == 2'd0) begin
                  rsp_payload_outputs_0 <= C_data_out[127:96];
                end else if (readC_counter == 2'd1) begin
                  rsp_payload_outputs_0 <= C_data_out[95:64];
                end else if (readC_counter == 2'd2) begin
                  rsp_payload_outputs_0 <= C_data_out[63:32];
                end else if (readC_counter == 2'd3) begin
                  rsp_payload_outputs_0 <= C_data_out[31:0];
                  C_index_forRead <= C_index_forRead + 1;
                end
                readC_counter <= readC_counter + 1;
              end
              7'd6:begin
                offset <= cmd_payload_inputs_0[31:0];
              end
              7'd10:begin
                relu_input_offset <=  cmd_payload_inputs_0[31:0];
                relu_output_offset <=  cmd_payload_inputs_1[31:0];
              end
              7'd11:begin
                relu_multiplier_id <=  cmd_payload_inputs_0[31:0];
                relu_multiplier_alpla <=  cmd_payload_inputs_1[31:0];
              end
              7'd12:begin
                relu_shift_id <=  cmd_payload_inputs_0[31:0];
                relu_shift_alpla <=  cmd_payload_inputs_1[31:0];
              end
              7'd13:begin
                relu_input <=  cmd_payload_inputs_0[31:0];
              end
              7'd14:begin
                // PATCH: Do not valid immediately. Wait for pipeline.
                rsp_valid <= 1'b0;
                relu_latency_cnt <= 0;
                cfu_state <= WAIT_RELU;
              end
              7'd15:begin
                relu_min <=  cmd_payload_inputs_0[31:0];
                relu_max <=  cmd_payload_inputs_1[31:0];
              end
            endcase
          end
        end

        // PATCH: New state to wait for 3-cycle ReLU pipeline
        WAIT_RELU: begin
            if (relu_latency_cnt == 2'd3) begin
                rsp_payload_outputs_0 <= relu_clamped_output;
                rsp_valid <= 1'b1;
                cfu_state <= WAIT_RSP;
            end else begin
                relu_latency_cnt <= relu_latency_cnt + 1;
            end
        end

        WAIT_RSP:begin
          if (rsp_valid && rsp_ready) begin 
            cmd_ready <= 1'b1;
            rsp_valid <= 1'b0;
            cfu_state <= READ_CFU;            
          end
        end

        default: cfu_state <= READ_CFU;
      endcase
    end
  end

endmodule



module leaky_relu(
  input               clk,
  input               reset,
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
  reg signed [31:0] s1_val_shift;
  reg signed [31:0] s1_multiplier;
  reg signed [31:0] s1_output_offset, s1_min, s1_max;
  reg signed [5:0]  s1_right_shift;

  // Moved from block-local temps to module-scope wires (Verilog-safe)
  wire signed [31:0] s1_val_comb = input_data - input_offset;
  wire signed [5:0]  s1_raw_shift_comb = s1_val_comb[31] ? $signed(shift_alpla[5:0]) : $signed(shift_id[5:0]);
  wire signed [31:0] s1_multiplier_comb = s1_val_comb[31] ? multiplier_alpla : multiplier_id;

  always @(posedge clk) begin
    if (reset) begin
      s1_val_shift      <= 0;
      s1_multiplier     <= 0;
      s1_right_shift    <= 0;
      s1_output_offset  <= 0;
      s1_min            <= 0;
      s1_max            <= 0;
    end else begin
      s1_multiplier <= s1_multiplier_comb;

      if (s1_raw_shift_comb > 0) begin
        s1_val_shift   <= (s1_val_comb <<< s1_raw_shift_comb);
        s1_right_shift <= 0;
      end else begin
        s1_val_shift   <= s1_val_comb;
        s1_right_shift <= -s1_raw_shift_comb;
      end

      s1_output_offset <= output_offset;
      s1_min           <= min;
      s1_max           <= max;
    end
  end

  // --- Pipeline Stage 2: Multiplication & Accumulation ---
  reg signed [31:0] s2_high_mul_result;
  reg signed [5:0]  s2_right_shift;
  reg signed [31:0] s2_output_offset, s2_min, s2_max;

  // Moved from block-local temps to module-scope wires (Verilog-safe)
  wire signed [63:0] s2_product_comb = $signed(s1_val_shift) * $signed(s1_multiplier);
  wire signed [63:0] s2_nudge_comb   = (s2_product_comb >= 0) ? 64'sd1073741824 : -64'sd1073741823; // +2^30 or (1-2^30)
  wire signed [63:0] s2_acc_comb     = s2_product_comb + s2_nudge_comb;
  wire signed [31:0] s2_high_mul_calc =
    (s2_acc_comb < 0 && s2_acc_comb[30:0] != 0) ? ($signed(s2_acc_comb[62:31]) + 1) : $signed(s2_acc_comb[62:31]);

  always @(posedge clk) begin
    if (reset) begin
      s2_high_mul_result <= 0;
      s2_right_shift     <= 0;
      s2_output_offset   <= 0;
      s2_min             <= 0;
      s2_max             <= 0;
    end else begin
      if (s1_val_shift == 32'sh8000_0000 && s1_multiplier == 32'sh8000_0000) begin
        s2_high_mul_result <= 32'sd2147483647;
      end else begin
        s2_high_mul_result <= s2_high_mul_calc;
      end

      s2_right_shift   <= s1_right_shift;
      s2_output_offset <= s1_output_offset;
      s2_min           <= s1_min;
      s2_max           <= s1_max;
    end
  end

  // --- Pipeline Stage 3: Scaling & Clamping ---
  // Moved from block-local temps to module-scope wires (Verilog-safe)
  wire [31:0] s3_mask =
    (s2_right_shift == 0) ? 32'd0 : ((32'h1 << s2_right_shift) - 1);

  wire [31:0] s3_remainder = (s2_high_mul_result[31:0] & s3_mask);

  wire [31:0] s3_threshold =
    (s3_mask >> 1) + ((s2_high_mul_result < 0) ? 32'd1 : 32'd0);

  wire signed [31:0] s3_final_scaled_val =
    ($signed(s2_high_mul_result >>> s2_right_shift)) + ((s3_remainder > s3_threshold) ? 32'sd1 : 32'sd0);

  wire signed [31:0] s3_unclamped = s2_output_offset + s3_final_scaled_val;

  always @(posedge clk) begin
    if (reset) begin
      clamped_output <= 0;
    end else begin
      if (s3_unclamped > s2_max) begin
        clamped_output <= s2_max;
      end else if (s3_unclamped < s2_min) begin
        clamped_output <= s2_min;
      end else begin
        clamped_output <= s3_unclamped;
      end
    end
  end

endmodule


module TPU(
    clk,
    reset,

    in_valid,
    K,
    M,
    N,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_wr_en,
    C_index,
    C_data_in,
    C_data_out,
    offset,
    cur_state,
    // K_num,
    // M_num,
    // N_num,
    // MdivFour,
    // NdivFour,
    // k,
    // m,
    // n, 
    C_write_counter 
    // C_reg_state,
    // see_sa_out0, 
    // see_sa_out1, 
    // see_sa_out2, 
    // see_sa_out3, 
    // see_a_out0,
    // see_a_out1, 
    // see_b_out0,
    // see_b_out1
);


input clk;
input reset;
input            in_valid;
input [7:0]      K;
input [7:0]      M;
input [7:0]      N;
output  reg      busy;

output           A_wr_en;
output reg [11:0]    A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output reg [11:0]    B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output reg       C_wr_en;
output reg [11:0]    C_index;
output reg [127:0]   C_data_in;
input  [127:0]   C_data_out;
input [31:0]     offset;

localparam IDLE = 3'd0;
localparam WORK = 3'd1;
// localparam WAIT = 3'd2;
localparam WRITE = 3'd2;
localparam DONE = 3'd3;
localparam ERROR = 3'd4;
output reg [2:0] cur_state; // set to output for cfu
reg pe_rst;

assign A_wr_en = 0;
assign B_wr_en = 0;
assign A_data_in = 32'd0;
assign B_data_in = 32'd0;

reg [7:0] MdivFour, NdivFour;
reg [7:0] K_num, M_num, N_num;
always @(posedge clk) begin
    if (reset) begin
        MdivFour <= 0;
        NdivFour <= 0;
        K_num <= 0;
        M_num <= 0;
        N_num <= 0;
    end else if (in_valid) begin
        K_num = K;
        M_num = M;
        N_num = N;
        if (M_num <= 4) begin
            MdivFour = 1;
        end else if (M_num[1:0] == 2'b00) begin
            MdivFour = M_num/4;
        end else begin
            MdivFour = M_num/4 + 1;
        end
        if (N_num <= 4) begin
            NdivFour = 1;
        end else if (N_num[1:0] == 2'b00) begin
            NdivFour = N_num/4;
        end else begin
            NdivFour = N_num/4 + 1;
        end
    end

end

//* Implement your design here
reg [8:0] A_in [3:0];
reg [7:0] B_in [3:0];
wire [127:0] C_sa_out [3:0];
// assign {C_sa_out[3], C_sa_out[2], C_sa_out[1], C_sa_out[0]} <= {0, 0, 0, 0};

reg [31:0] A_reg [3:0];
reg [31:0] B_reg [3:0];
reg [127:0] C_reg [3:0];

// reg_state control which reg (0, 1, 2, 3) should give data to A_in (0, 1, 2, 3)
reg [1:0] A_reg_state [3:0];
reg [1:0] B_reg_state [3:0];
reg [2:0] C_reg_state;

reg [7:0] m, k, n;
// reg [15:0] restore_A;
// reg [15:0] restore_B;
// reg [15:0] A_addr_cur;
// reg [15:0] B_addr_cur;
// reg [15:0] C_addr_cur;
// reg [1:0] C_sa_index;
reg [2:0] wait_cycles;
output reg [2:0] C_write_counter;
reg [8:0] offset_inTPU;
// reg [127:0] see_sa_out0, see_sa_out1, see_sa_out2, see_sa_out3;
// reg [31:0] see_a_out0, see_a_out1, see_b_out0, see_b_out1;
// reg [3:0] control_see_counter;
// reg [2:0] control_ab_counter;
always @(posedge clk) begin
    if (reset) begin
        pe_rst <= 1;
        cur_state <= IDLE;
        m <= 0;
        k <= 0;
        n <= 0;
        A_index <= 0;
        B_index <= 0;
        C_index <= 0;
        // restore_A <= 0;
        // restore_B <= 0;
        // A_addr_cur <= 0;
        // B_addr_cur <= 0;
        // C_addr_cur <= 0;
        // C_sa_index <= 0;
        wait_cycles <= 0;
        C_write_counter <= 0;
        busy <= 0;

        A_in[0] <= 0;
        A_in[1] <= 0;
        A_in[2] <= 0;
        A_in[3] <= 0;
        B_in[0] <= 0;
        B_in[1] <= 0;
        B_in[2] <= 0;
        B_in[3] <= 0;

        A_reg[0] <= 32'd0;
        A_reg[1] <= 32'd0;
        A_reg[2] <= 32'd0;
        A_reg[3] <= 32'd0;
        B_reg[0] <= 32'd0;
        B_reg[1] <= 32'd0;
        B_reg[2] <= 32'd0;
        B_reg[3] <= 32'd0;
        C_reg[0] <= 128'd0;
        C_reg[1] <= 128'd0;
        C_reg[2] <= 128'd0;
        C_reg[3] <= 128'd0;

        A_reg_state[0] <= 0;
        A_reg_state[1] <= 0;
        A_reg_state[2] <= 0;
        A_reg_state[3] <= 0;
        B_reg_state[0] <= 0;
        B_reg_state[1] <= 0;
        B_reg_state[2] <= 0;
        B_reg_state[3] <= 0;
        C_reg_state <= 3'd0;

        // see_sa_out0 <= 0;
        // see_sa_out1 <= 0;
        // see_sa_out2 <= 0;
        // see_sa_out3 <= 0;
        // control_see_counter <= 0;
        // control_ab_counter <= 0;
        // see_a_out0 <= 0;
        // see_a_out1 <= 0;
        // see_b_out0 <= 0;
        // see_b_out1 <= 0;
    end else if (in_valid) begin
        pe_rst <= 1;
        cur_state <= IDLE;
        m <= 0;
        k <= 0;
        n <= 0;
        A_index <= 0;
        B_index <= 0;
        C_index <= 0;
        // restore_A <= 0;
        // restore_B <= 0;
        // A_addr_cur <= 0;
        // B_addr_cur <= 0;
        // C_addr_cur <= 0;
        // C_sa_index <= 0;
        C_write_counter <= 0;
        busy <= 1;

        A_reg[0] <= 32'd0;
        A_reg[1] <= 32'd0;
        A_reg[2] <= 32'd0;
        A_reg[3] <= 32'd0;
        B_reg[0] <= 32'd0;
        B_reg[1] <= 32'd0;
        B_reg[2] <= 32'd0;
        B_reg[3] <= 32'd0;
        C_reg[0] <= 128'd0;
        C_reg[1] <= 128'd0;
        C_reg[2] <= 128'd0;
        C_reg[3] <= 128'd0;

        A_reg_state[0] <= 2'd0;
        A_reg_state[1] <= 2'd3;
        A_reg_state[2] <= 2'd2;
        A_reg_state[3] <= 2'd1;
        B_reg_state[0] <= 2'd0;
        B_reg_state[1] <= 2'd3;
        B_reg_state[2] <= 2'd2;
        B_reg_state[3] <= 2'd1;
        C_reg_state <= 3'd0;
        offset_inTPU <= offset[8:0];
        // see_sa_out0 <= 0;
        // see_sa_out1 <= 0;
        // see_sa_out2 <= 0;
        // see_sa_out3 <= 0;
        // control_see_counter <= 0;
        // control_ab_counter <= 0;
        // see_a_out0 <= 0;
        // see_a_out1 <= 0;
        // see_b_out0 <= 0;
        // see_b_out1 <= 0;
    end else if (busy) begin
        case (cur_state)
            IDLE: begin
                if (busy) begin
                    cur_state <= WORK;
                end else begin
                    cur_state <= IDLE;
                end
            end

            WORK: begin
                pe_rst <= 0;
                C_wr_en <= 0; // switch to WORK: 1 cycle, next cycle: C_wr_en = 0

                // if (control_ab_counter == 3'd0) begin
                //   see_a_out0 <= A_data_out;
                //   see_b_out0 <= B_data_out;
                //   control_ab_counter <= control_ab_counter + 1;
                // end else if (control_ab_counter == 3'd1) begin
                //   see_a_out1 <= A_data_out;
                //   see_b_out1 <= B_data_out;
                //   control_ab_counter <= control_ab_counter + 1;
                // end

                A_reg[A_reg_state[0]] = k < K_num? $signed(A_data_out) : 0;
                A_in[0] <= (k < K_num) ? $signed(A_data_out[31:24]) + $signed(offset): 0; // initial: A_reg(0 -> 1 -> 2 -> 3 -> 0 -> ...)
                A_in[1] <= (k < K_num + 1) ? $signed(A_reg[A_reg_state[1]][23:16]) + $signed(offset): 0; // initial: A_reg(3 -> 0 -> 1 -> 2 -> 3 -> ...)
                A_in[2] <= (k < K_num + 2) ? $signed(A_reg[A_reg_state[2]][15:8]) + $signed(offset): 0; // initial: A_reg(2 -> 3 -> 0 -> 1 -> 2 -> ...)
                A_in[3] <= (k < K_num + 3) ? $signed(A_reg[A_reg_state[3]][7:0]) + $signed(offset): 0; // initial: A_reg(1 -> 2 -> 3 -> 0 -> 1 -> ...)

                B_reg[B_reg_state[0]] = k < K_num? $signed(B_data_out) : 0;
                B_in[0] <= (k < K_num) ? $signed(B_data_out[31:24]) : 0; // initial: B_reg(0 -> 1 -> 2 -> 3 -> 0 -> ...)
                B_in[1] <= (k < K_num + 1) ? $signed(B_reg[B_reg_state[1]][23:16]) : 0; // initial: B_reg(3 -> 0 -> 1 -> 2 -> 3 -> ...)
                B_in[2] <= (k < K_num + 2) ? $signed(B_reg[B_reg_state[2]][15:8]) : 0; // initial: B_reg(2 -> 3 -> 0 -> 1 -> 2 -> ...)
                B_in[3] <= (k < K_num + 3) ? $signed(B_reg[B_reg_state[3]][7:0]) : 0; // initial: B_reg(1 -> 2 -> 3 -> 0 -> 1 -> ...)
                    
                // {A_reg_state[0], A_reg_state[1], A_reg_state[2], A_reg_state[3]} <= {A_reg_state[3], A_reg_state[0], A_reg_state[1], A_reg_state[2]};
                A_reg_state[0] <= A_reg_state[3];
                A_reg_state[1] <= A_reg_state[0];
                A_reg_state[2] <= A_reg_state[1];
                A_reg_state[3] <= A_reg_state[2];
                // {B_reg_state[0], B_reg_state[1], B_reg_state[2], B_reg_state[3]} <= {B_reg_state[3], B_reg_state[0], B_reg_state[1], B_reg_state[2]};
                B_reg_state[0] <= B_reg_state[3];
                B_reg_state[1] <= B_reg_state[0];
                B_reg_state[2] <= B_reg_state[1];
                B_reg_state[3] <= B_reg_state[2];
                A_index <= m*K_num + k + 1;
                B_index <= n*K_num + k + 1;
                // if (k < K_num) begin
                //     restore_A = A_index;
                //     restore_B = B_index;
                // end
                k <= k + 1;

                if (k + 3 > K_num + 6) begin  // will switch to WRITE state at cycle = k + 2, then write at cycle = k + 3 ~ k + 6                
                    C_reg_state <= 2'd0;
                    C_wr_en <= 1;
                    cur_state <= WRITE;
                end 
            end 


            WRITE: begin
                // C_sa_index = C_reg_state[0];
                C_index = n*M_num + m*4 + C_write_counter;
                C_data_in = C_sa_out[C_write_counter];

                // if (control_see_counter == 4'd0) begin
                //   see_sa_out0 = C_sa_out[C_write_counter];
                //   control_see_counter <= control_see_counter + 1;
                // end else if (control_see_counter == 4'd1) begin
                //   see_sa_out1 = C_sa_out[C_write_counter];
                //   control_see_counter <= control_see_counter + 1;
                // end else if (control_see_counter == 4'd2) begin
                //   see_sa_out2 = C_sa_out[C_write_counter];
                //   control_see_counter <= control_see_counter + 1;
                // end else if (control_see_counter == 4'd3) begin
                //   see_sa_out3 = C_sa_out[C_write_counter];
                //   control_see_counter <= control_see_counter + 1;
                // end


                // {C_reg_state[0], C_reg_state[1], C_reg_state[2], C_reg_state[3]} <= {C_reg_state[3], C_reg_state[0], C_reg_state[1], C_reg_state[2]};

                // if (C_reg_state == 2'd0) begin
                //   C_reg_state <= 2'd1;
                // end if (C_reg_state == 2'd1) begin
                //   C_reg_state <= 2'd2;
                // end if (C_reg_state == 2'd2) begin
                //   C_reg_state <= 2'd3;
                // end if (C_reg_state == 2'd3) begin
                //   C_reg_state <= 2'd0;
                // end
                // C_reg_state <= C_reg_state + 1;
                
                if (m+1 == MdivFour) begin // m stuck at 3, n stuck at 0
                    if (m*4 + C_write_counter >= M_num - 1) begin // 3 * 4 + () >= 15

                        if (n+1> NdivFour) begin
                            cur_state <= DONE;
                        end else begin
                            pe_rst <= 1;
                            k <= 0;
                            m <= 0;
                            n <= n + 1;

                            A_index <= 0;
                            B_index <= (n+1)*K_num;
                            A_in[0] <= 0;
                            A_in[1] <= 0;
                            A_in[2] <= 0;
                            A_in[3] <= 0;
                            B_in[0] <= 0;
                            B_in[1] <= 0;
                            B_in[2] <= 0;
                            B_in[3] <= 0;

                            A_reg[0] <= 32'd0;
                            A_reg[1] <= 32'd0;
                            A_reg[2] <= 32'd0;
                            A_reg[3] <= 32'd0;
                            B_reg[0] <= 32'd0;
                            B_reg[1] <= 32'd0;
                            B_reg[2] <= 32'd0;
                            B_reg[3] <= 32'd0;

                            A_reg_state[0] <= 2'd0;
                            A_reg_state[1] <= 2'd3;
                            A_reg_state[2] <= 2'd2;
                            A_reg_state[3] <= 2'd1;
                            B_reg_state[0] <= 2'd0;
                            B_reg_state[1] <= 2'd3;
                            B_reg_state[2] <= 2'd2;
                            B_reg_state[3] <= 2'd1;
                            C_write_counter <= 0;
                            cur_state <= WORK;
                        end
                    end else if (C_write_counter <= 4) begin
                        C_write_counter <= C_write_counter + 1; 
                    end else if (C_write_counter > 4) begin
                        cur_state <= ERROR;
                    end
                    
                end else begin // m + 1 < MdivFour
                    if (C_write_counter == 3) begin
                        pe_rst <= 1;
                        k <= 0;
                        n <= n;
                        m <= m + 1;

                        A_index <= (m+1)*K_num;
                        B_index <= n*K_num;
                        A_in[0] <= 0;
                        A_in[1] <= 0;
                        A_in[2] <= 0;
                        A_in[3] <= 0;
                        B_in[0] <= 0;
                        B_in[1] <= 0;
                        B_in[2] <= 0;
                        B_in[3] <= 0;

                        A_reg[0] <= 32'd0;
                        A_reg[1] <= 32'd0;
                        A_reg[2] <= 32'd0;
                        A_reg[3] <= 32'd0;
                        B_reg[0] <= 32'd0;
                        B_reg[1] <= 32'd0;
                        B_reg[2] <= 32'd0;
                        B_reg[3] <= 32'd0;

                        A_reg_state[0] <= 2'd0;
                        A_reg_state[1] <= 2'd3;
                        A_reg_state[2] <= 2'd2;
                        A_reg_state[3] <= 2'd1;
                        B_reg_state[0] <= 2'd0;
                        B_reg_state[1] <= 2'd3;
                        B_reg_state[2] <= 2'd2;
                        B_reg_state[3] <= 2'd1;
                        
                        C_write_counter <= 0;
                        
                        cur_state <= WORK;
                    end else if (C_write_counter < 3) begin
                        C_write_counter <= C_write_counter + 1;
                    end else if (C_write_counter > 3) begin
                        cur_state <= ERROR;
                    end
                end 
                
            end

            DONE: begin
                C_wr_en <= 0;
                busy <= 1'd0;
                cur_state <= IDLE;
            end

            ERROR: begin
              cur_state <= ERROR;
            end

            default: cur_state <= IDLE;
        endcase
    end

end

// always @(posedge clk) begin
//       $display("State: %0d, in_valid: %0d, write_counter: %0d, k: %0d, m: %0d, n: %0d, K_num: %0d, M_num: %0d, N_num: %0d, Mdiv4: %0d, Ndiv4: %0d, A_index: %0x, B_index: %0x, C_index: %0d, C_reg_state[0]: %0x, A_in: %0x %0x %0x %0x, B_in: %0x %0x %0x %0x, A_reg: %0x %0x %0x %0x, B_reg: %0x %0x %0x %0x, A_data_out: %0x, B_data_out: %0x, C_data_in: %0x", 
//           cur_state, in_valid, C_write_counter, k, m, n, K_num, M_num, N_num, MdivFour, NdivFour, 
//           A_index, B_index, C_index, C_reg_state[0], 
//           A_in[0], A_in[1], A_in[2], A_in[3], 
//           B_in[0], B_in[1], B_in[2], B_in[3], 
//           A_reg[0], A_reg[1], A_reg[2], A_reg[3],
//           B_reg[0], B_reg[1], B_reg[2], B_reg[3],
//           A_data_out, B_data_out,
//           C_data_in);
//       $display("C_sa_out: %0x %0x %0x %0x", C_sa_out[0], C_sa_out[1], C_sa_out[2], C_sa_out[3]);
//     end

Systolic_array sa(
    .clk(clk),
    .reset(reset),
    .pe_rst(pe_rst),
    .left_0(A_in[0]),
    .left_1(A_in[1]),
    .left_2(A_in[2]), 
    .left_3(A_in[3]),
    .top_0(B_in[0]),
    .top_1(B_in[1]),
    .top_2(B_in[2]),
    .top_3(B_in[3]),
    // .offset(offset),
    // .C_reg_state(C_sa_index),
    .out_0(C_sa_out[0]),
    .out_1(C_sa_out[1]),
    .out_2(C_sa_out[2]),
    .out_3(C_sa_out[3])
);

endmodule


module Systolic_array(clk, reset, pe_rst, left_0, left_1, left_2, left_3, top_0, top_1, top_2, top_3, out_0, out_1, out_2, out_3);

input clk;
input reset;
input pe_rst;
input signed [8:0] left_0, left_1, left_2, left_3;
input signed [7:0] top_0, top_1, top_2, top_3;
// input [31:0] offset;
output reg [127:0] out_0, out_1, out_2, out_3;

parameter ARRAY_SIZE = 4;
// wire [7:0] left_reg [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
// wire [7:0] top_reg [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
wire [8:0] right_reg [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
wire [7:0] bottom_reg [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
wire [31:0] acc_reg [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];


genvar i, j;
generate
    for (i = 0; i < 4; i = i + 1) begin 
        for (j = 0; j < 4; j = j + 1) begin 
            PE pe (
                .clk(clk),
                .reset(reset),
                .pe_rst(pe_rst),
                // .offset(offset),
                .left_in((j == 0) ? (i == 0 ? left_0 : (i == 1 ? left_1 : (i == 2 ? left_2 : left_3))) : right_reg[i][j-1]),
                .top_in((i == 0) ? (j == 0 ? top_0 : (j == 1 ? top_1 : (j == 2 ? top_2 : top_3))) : bottom_reg[i-1][j]),
                .right_out(right_reg[i][j]),
                .bottom_out(bottom_reg[i][j]),
                .acc(acc_reg[i][j])
            );
        end
    end
endgenerate

// output to C
always @(posedge clk) begin
    out_0 = {acc_reg[0][0], acc_reg[0][1], acc_reg[0][2], acc_reg[0][3]};
    out_1 = {acc_reg[1][0], acc_reg[1][1], acc_reg[1][2], acc_reg[1][3]};
    out_2 = {acc_reg[2][0], acc_reg[2][1], acc_reg[2][2], acc_reg[2][3]};
    out_3 = {acc_reg[3][0], acc_reg[3][1], acc_reg[3][2], acc_reg[3][3]};
        
end

endmodule


module PE(
    clk, 
    reset,
    pe_rst, 
    left_in, 
    top_in, 
    right_out, 
    bottom_out, 
    acc);

input clk;
input reset;
input pe_rst;
// input [31:0] offset;
input [8:0] left_in;
input [7:0] top_in;
output reg [8:0] right_out;
output reg [7:0] bottom_out;
output reg [31:0] acc;
// reg [8:0] left_in_add_offset;

    always @(posedge clk) begin
        if (reset) begin
            right_out <= 9'd0;
            bottom_out <= 8'd0;
            acc <= 32'd0;
        end else if (pe_rst) begin
            acc <= 32'd0;
        end else begin
            // left_in_add_offset = $signed(left_in) + $signed(offset[8:0]);
            acc <= ($signed(left_in) * $signed(top_in)) + $signed(acc);
            // acc <= (left_in * top_in) + acc;
            right_out <= left_in;
            bottom_out <= top_in;
        end
    end
endmodule


// global_buffer_bram is for using BRAM to integrate systolic array
module global_buffer_bram #(parameter ADDR_BITS=8, parameter DATA_BITS=8)(
  input                      clk,
  input                      rst_n,
  input                      ram_en,
  input                      wr_en,
  input      [ADDR_BITS-1:0] index,
  input      [DATA_BITS-1:0] data_in,
  output reg [DATA_BITS-1:0] data_out
  );

  parameter DEPTH = 2**ADDR_BITS;

  reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

  always @ (negedge clk) begin
    if (ram_en) begin
      if(wr_en) begin
        gbuff[index] <= data_in;
      end else begin
        data_out <= gbuff[index];
      end
    end
  end

endmodule

/**
  Example of instantiating a global_buffer_bram: 

  global_buffer_bram #(
    .ADDR_BITS(12), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_A(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(A_wr_en),
    .index(A_index),
    .data_in(A_data_in),
    .data_out(A_data_out)
  );

*/