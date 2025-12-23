// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



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

  // Trivial handshaking for a combinational CFU
  // assign rsp_valid = cmd_valid;
  // assign cmd_ready = rsp_ready;

  //
  // select output -- note that we're not fully decoding the 3 function_id bits
  //
  // assign rsp_payload_outputs_0 = cmd_payload_function_id[0] ? 
  //                                          cmd_payload_inputs_1 :
  //                                          cmd_payload_inputs_0 ;

  reg             in_valid;
  reg [15:0]       K;
  reg [15:0]       M;
  reg [15:0]       N;
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
  wire [2:0] cur_state_fromTPU; // for testing tpu state
  reg [1:0] readC_counter;

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

  localparam READ_CFU = 2'd0;
  localparam WAIT_RSP = 2'd1;
  reg [1:0] cfu_state;
  wire check_CFU_or_TPU;
  assign check_CFU_or_TPU = busy;
  
  // wire [7:0] K_num, M_num, N_num;
  // wire [7:0] MdivFour, NdivFour;
  // wire [7:0] k, m, n;
  wire [2:0] C_write_counter;
  // wire [2:0] C_reg_state;

  // reg [31:0] cfu_counter;
  // reg print;
  // wire [127:0] see_sa_out0, see_sa_out1, see_sa_out2, see_sa_out3;
  // wire [31:0] see_a_out0, see_a_out1, see_b_out0, see_b_out1;

  // assign cmd_ready = ~rsp_valid;
  assign A_index = check_CFU_or_TPU ? A_index_forRead : (print ? A_index_forPrint: A_index_forWrite);
  assign A_wr_en = check_CFU_or_TPU ? 0 : A_wr_en_fromCFU;
  assign B_index = check_CFU_or_TPU ? B_index_forRead : (print ? B_index_forPrint: B_index_forWrite);
  assign B_wr_en = check_CFU_or_TPU ? 0 : B_wr_en_fromCFU;
  assign C_index = check_CFU_or_TPU ? C_index_forWrite : C_index_forRead;
  assign C_wr_en = check_CFU_or_TPU ? C_wr_en_fromTPU : C_wr_en_fromCFU;
  global_buffer_bram #(
    .ADDR_BITS(16), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_A(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(A_wr_en),
    .index(A_index),
    .data_in(A_data_in),
    .data_out(A_data_out) // out
  );

  global_buffer_bram #(
    .ADDR_BITS(16), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_B(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(B_wr_en),
    .index(B_index),
    .data_in(B_data_in),
    .data_out(B_data_out) // out
  );
  global_buffer_bram #(
    .ADDR_BITS(9), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(128)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_C(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(C_wr_en),
    .index(C_index),
    .data_in(C_data_in),
    .data_out(C_data_out) // out
  );
  
  TPU My_TPU(
    .clk            (clk),  
    .reset          (reset),       
    .in_valid       (in_valid),         
    .K              (K), 
    .M              (M), 
    .N              (N), 
    .busy           (busy),         // out
    .A_wr_en        (),      // out   
    .A_index        (A_index_forRead),      // out   
    .A_data_in      (),    // out      
    .A_data_out     (A_data_out),         
    .B_wr_en        (),      // out   
    .B_index        (B_index_forRead),      // out    
    .B_data_in      (),    // out     
    .B_data_out     (B_data_out),         
    .C_wr_en        (C_wr_en_fromTPU),      // out   
    .C_index        (C_index_forWrite),      // out   
    .C_data_in      (C_data_in),    // out     
    .C_data_out     (C_data_out),
    .offset         (offset)
  );

  leaky_relu My_relu(
    .input_data (relu_input),
    .input_offset(relu_input_offset),
    .output_offset(relu_output_offset),
    .multiplier_id(relu_multiplier_id),
    .multiplier_alpla(relu_multiplier_alpla),
    .shift_id(relu_shift_id),
    .shift_alpla(relu_shift_alpla),
    .min(relu_min),
    .max(relu_max),
    .clamped_output(relu_clamped_output)
  );


  
  always @(posedge clk) begin
    // cfu_counter <= cfu_counter + 1;
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

      // cfu_counter <= 0;
      print <= 0;
    
    end else begin
      case (cfu_state)
        READ_CFU: begin
          if (cmd_valid && cmd_ready) begin
            cmd_ready <= 1'b0;
            rsp_valid <= 1'b1;
            cfu_state <= WAIT_RSP;
            case(cmd_payload_function_id[9:3])
              7'd0:begin
                A_wr_en_fromCFU <= 1'b0;
                B_wr_en_fromCFU <= 1'b0;
                A_index_forWrite <= 12'd0 - 12'd1; // -1
                B_index_forWrite <= 12'd0 - 12'd1; // -1
                print = 0;
              end
              7'd1:begin
                A_wr_en_fromCFU = 1'b1;
                // B_wr_en_fromCFU = 1'b1;
                A_index_forWrite <= A_index_forWrite + 1; 
                // B_index_forWrite <= B_index_forWrite + 1;
                A_data_in <= cmd_payload_inputs_0[31:0];
                // B_data_in <= cmd_payload_inputs_1[31:0];
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
                end else begin // busy == 0
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
                rsp_payload_outputs_0 <=  relu_clamped_output;
              end
              7'd15:begin
                relu_min <=  cmd_payload_inputs_0[31:0];
                relu_max <=  cmd_payload_inputs_1[31:0];
              end
              
            endcase
            /*
            else if (cmd_payload_function_id[9:3] == 7'd6) begin
              offset <= cmd_payload_inputs_0[31:0];
            end else if (cmd_payload_function_id[9:3] == 7'd7) begin 
              rsp_payload_outputs_0 <= {8'b0, K, M, N};
            end else if (cmd_payload_function_id[9:3] == 7'd8) begin
              rsp_payload_outputs_0 <= cur_state_fromTPU;

            end else if (cmd_payload_function_id[9:3] == 7'd9) begin
              rsp_payload_outputs_0 <= C_write_counter;
            end //else if (cmd_payload_function_id[9:3] == 7'd10) begin
            //   // rsp_payload_outputs_0 <= see_sa_out2;
            //   rsp_payload_outputs_0 <= see_b_out0;
            // end else if (cmd_payload_function_id[9:3] == 7'd11) begin
            //   // rsp_payload_outputs_0 <= see_sa_out3;
            //   rsp_payload_outputs_0 <= see_b_out1;
            // end
            else if (cmd_payload_function_id[9:3] == 7'd10) begin
              case(cmd_payload_inputs_1[1:0])
                2'b00:rsp_payload_outputs_0 <= {cmd_payload_inputs_0[7:0],24'b0};
                2'b01:rsp_payload_outputs_0[23:16] <= cmd_payload_inputs_0[7:0];
                2'b10:rsp_payload_outputs_0[15:8] <= cmd_payload_inputs_0[7:0];
                2'b11:rsp_payload_outputs_0[7:0] <= cmd_payload_inputs_0[7:0];
              endcase
              
            end 
            */
          end
        end

        WAIT_RSP:begin
          if (rsp_valid && rsp_ready) begin // when rsp_valid is 1, it will wait rsp_ready is 0 to handshake the 
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
  input_data ,
  input_offset,
  output_offset,
  multiplier_id,
  multiplier_alpla,
  shift_id,
  shift_alpla,
  min,
  max,
  clamped_output
);

  input signed [31:0] input_data;
  input signed  [31:0] input_offset;
  input signed  [31:0] output_offset;
  input signed  [31:0] multiplier_id;
  input signed  [31:0] multiplier_alpla;
  input signed  [31:0] shift_id;
  input signed  [31:0] shift_alpla;
  input signed  [31:0] min;
  input signed  [31:0] max;
  output reg signed  [31:0] clamped_output;

  wire signed [31:0] val;
  assign val = input_data - input_offset;

  reg signed [31:0] multiplier;
  reg signed [ 5:0] shift;
  
  always@(*)begin
    if(val[31])begin
      multiplier = multiplier_alpla;
      shift = shift_alpla[5:0];
    end
    else begin
      multiplier = multiplier_id;
      shift = shift_id[5:0];
    end
  end
    
  
  reg signed [31:0] val_shift;
  reg signed [ 5:0] right_shift;

  always@(*)begin
    if($signed(shift) > 0)begin
      val_shift = val <<< shift;
      right_shift = 0;
    end
    else begin
      val_shift = val;
      right_shift = -$signed(shift);
    end

  end
    

  reg signed [63:0] nudge;
  reg signed [63:0] product;
  reg signed [63:0] acc;
  reg signed [31:0] high_mul_result; 
    

  always@(*)begin
    product = val_shift * multiplier;
    nudge = (product >=0)? (1 << 30) : (1 - (1 << 30));
    acc = product + nudge;
    if( val_shift ==  -2147483648 && multiplier ==  -2147483648)begin
      high_mul_result = 2147483647;
    end
    else begin
      high_mul_result = (acc <0 && acc[30:0] != 0) ? acc[62:31] +1 : acc[62:31];
    end
  end
  
  reg signed [31:0] final_scaled_val;
  reg [31:0] mask;
  reg [31:0] remainder;
  reg [31:0] threshold;

  always@(*)begin
    mask = (1 << right_shift) - 1;
    remainder = high_mul_result & mask;
    threshold = (mask >> 1) + (high_mul_result < 0 ? 1 : 0);

    final_scaled_val = (high_mul_result >>> right_shift) + (remainder > threshold ? 1 : 0);
  end

  reg signed [31:0] unclamped;
  
  always@(*)begin
    unclamped = output_offset + final_scaled_val;

  end

  always @(*) begin
      if (unclamped > max) begin
          clamped_output = max;
      end else if (unclamped < min) begin
          clamped_output = min;
      end else begin
          clamped_output = unclamped;
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
    offset
);


input clk;
input reset;
input            in_valid;
input [15:0]      K;
input [15:0]      M;
input [15:0]      N;
output  reg      busy;

output           A_wr_en;
output reg [15:0]    A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output reg [15:0]    B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output reg       C_wr_en;
output reg [8:0]    C_index;
output reg [127:0]   C_data_in;
input  [127:0]   C_data_out;
input [31:0]     offset;

localparam IDLE = 3'd0;
localparam WORK = 3'd1;
// localparam WAIT = 3'd2;
localparam WRITE = 3'd2;
localparam DONE = 3'd3;
localparam ERROR = 3'd4;
reg [2:0] cur_state; // set to output for cfu
reg pe_rst;

assign A_wr_en = 0;
assign B_wr_en = 0;
assign A_data_in = 32'd0;
assign B_data_in = 32'd0;

reg [15:0] MdivFour, NdivFour;
reg [15:0] K_num, M_num, N_num;
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
reg [2:0] C_write_counter;
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