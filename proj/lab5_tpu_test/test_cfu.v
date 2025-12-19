//need modify
`include "RTL/TPU.v"
`include "global_buffer_bram.v"
module Cfu #(  
    parameter ADDR_BITS=12,
    parameter DATA_BITS=32,
    parameter C_BITS=128,
    parameter S0 = 4'b0000,
    parameter S1 = 4'b0001,
    parameter S2 = 4'b0010,
    parameter S3 = 4'b0011,
    parameter S4 = 4'b0100,
    parameter S5 = 4'b0101,
    parameter S6 = 4'b0110,
    parameter S7 = 4'b0111,
    parameter S8 = 4'b1000,
    parameter S9 = 4'b1001,
    parameter S10 = 4'b1010,
    parameter S11 = 4'b1011,
    parameter S12 = 4'b1100

)(
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);
  //define the wires that load the inputs into the TPU
  wire [ADDR_BITS-1:0] A_index,B_index,C_index;
  wire [31:0] A_data_in,A_data_out,B_data_in,B_data_out,C_data_in,C_data_out;
  wire [1:0] A_wr_en,B_wr_en,C_wr_en,A_wr_mux_en,B_wr_mux_en,C_wr_mux_en;
  wire [ADDR_BITS-1:0] A_index_mux , B_index_mux , C_index_mux;
  reg [ADDR_BITS-1:0] A_index_init, B_index_init, C_index_init;
  reg [31:0] A_data_in_init, B_data_in_init;
  reg [C_BITS-1:0] C_data_in_init;
  
  reg A_wr_en_init;
  reg B_wr_en_init;
  reg C_wr_en_init;

  reg rst_n;
  reg in_valid;
  reg [7:0] K, M, N;
  wire [6:0] op;
  assign op = cmd_payload_function_id[9:3]; 
  //we have multiplexer to handle the loading of data, we only want to write into C when TPU is active
  reg tpu_busy;
  wire busy;
  assign A_wr_en_mux = (in_valid | tpu_busy | busy) ? A_wr_en : A_wr_en_init;
  assign B_wr_en_mux = (in_valid | tpu_busy | busy) ? B_wr_en : B_wr_en_init;
  assign C_wr_en_mux = (tpu_busy | busy) ? C_wr_en : C_wr_en_init;

  assign A_index_mux = (in_valid | tpu_busy | busy) ? A_index : A_index_init;
  assign B_index_mux = (in_valid | tpu_busy | busy) ? B_index : B_index_init;
  assign C_index_mux = (tpu_busy | busy) ? C_index : C_index_init;

  assign A_data_in_mux = (in_valid | tpu_busy | busy) ? A_data_in : A_data_in_init;
  assign B_data_in_mux = (in_valid | tpu_busy | busy) ? B_data_in : B_data_in_init;
  assign C_data_in_mux = (tpu_busy | busy) ? C_data_in : C_data_in_init;
  //define the bram to store the data
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
  global_buffer_bram #(
    .ADDR_BITS(12), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
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
    .ADDR_BITS(12), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
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
    .rst_n          (reset),     
    .in_valid       (in_valid),         
    .K              (K), 
    .M              (M), 
    .N              (N), 
    .busy           (busy),     
    .A_wr_en        (A_wr_en),         
    .A_index        (A_index),         
    .A_data_in      (A_data_in),         
    .A_data_out     (A_data_out),         
    .B_wr_en        (B_wr_en),         
    .B_index        (B_index),         
    .B_data_in      (B_data_in),         
    .B_data_out     (B_data_out),         
    .C_wr_en        (C_wr_en),         
    .C_index        (C_index),         
    .C_data_in      (C_data_in),         
    .C_data_out     (C_data_out)         
);
  // Only not ready for a command when we have a response.
  assign cmd_ready = ~rsp_valid;
  always @(posedge clk) begin
    if (reset) begin
      tpu_busy <= 1'b0;
    end else begin
      tpu_busy <= busy;
    end
  end
  always @(posedge clk) begin
    if (reset) begin
      rsp_payload_outputs_0 <= 32'b0;
      rsp_valid <= 1'b0;
    end else if (rsp_valid) begin
      // Waiting to hand off response to CPU.
      rsp_valid <= ~rsp_ready;
    end else if (cmd_valid) begin
      rsp_valid <= 1'b1;
      // Accumulate step:
      case (op)
      //have 0 as reset all
        2'b000_0000: begin
           rst_n <= 1'b0;
            K = 'bx;
            M = 'bx;
            N = 'bx;
        end
      //have 1 as set K
        2'b000_0001: begin
          K<=cmd_payload_inputs_0;
        end
      //2 to set M
        2'd2: begin
          M<=cmd_payload_inputs_0;
        end
        2'd3: begin
          N<=cmd_payload_inputs_0;
        end
        //set buffer A
        2'd4: begin
          A_index_init <= cmd_payload_inputs_0[ADDR_BITS-1:0];
          A_data_in_init <= cmd_payload_inputs_1;
          A_wr_en_init <= 1'b1;
        end
        //set buffer B
        2'd5: begin
          B_index_init <= cmd_payload_inputs_0[ADDR_BITS-1:0];
          B_data_in_init <= cmd_payload_inputs_1;
          B_wr_en_init <= 1'b1;
        end
        //set in_valid
        2'd6: begin
          A_wr_en_init <= 1'b0;
          B_wr_en_init <= 1'b0;
          in_valid <= 1'b1;
          rsp_payload_outputs_0 <= busy;
        end
        //get buffer C
        2'd7: begin
          C_index_init <= cmd_payload_inputs_0[ADDR_BITS-1:0];
          C_wr_en_init <= 1'b0;
        end
        default: begin
          
        end
      endcase
    end
  end
endmodule