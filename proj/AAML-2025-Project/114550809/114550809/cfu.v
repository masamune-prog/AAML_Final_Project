//need modify

module Cfu (
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
  reg [8:0] InputOffset, FilterOffset;

  // SIMD multiply step:
  wire signed [16:0] prod_0, prod_1, prod_2, prod_3;
  assign prod_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + $signed(InputOffset))
                  * $signed(cmd_payload_inputs_1[7 : 0]);
  assign prod_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + $signed(InputOffset))
                  * $signed(cmd_payload_inputs_1[15: 8]);
  assign prod_2 =  ($signed(cmd_payload_inputs_0[23:16]) + $signed(InputOffset))
                  * $signed(cmd_payload_inputs_1[23:16]);
  assign prod_3 =  ($signed(cmd_payload_inputs_0[31:24]) + $signed(InputOffset))
                  * $signed(cmd_payload_inputs_1[31:24]);

  wire signed [31:0] sum_prods;
  assign sum_prods = prod_0 + prod_1 + prod_2 + prod_3;

  // SIMD multiply step:
  wire signed [16:0] prod_fc_0, prod_fc_1, prod_fc_2, prod_fc_3;
  assign prod_fc_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + $signed(InputOffset))
                    * ($signed(cmd_payload_inputs_1[7 : 0]) + $signed(FilterOffset));
  assign prod_fc_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + $signed(InputOffset))
                    * ($signed(cmd_payload_inputs_1[15: 8]) + $signed(FilterOffset));
  assign prod_fc_2 =  ($signed(cmd_payload_inputs_0[23:16]) + $signed(InputOffset))
                    * ($signed(cmd_payload_inputs_1[23:16]) + $signed(FilterOffset));
  assign prod_fc_3 =  ($signed(cmd_payload_inputs_0[31:24]) + $signed(InputOffset))
                    * ($signed(cmd_payload_inputs_1[31:24]) + $signed(FilterOffset));

  wire signed [31:0] sum_prods_fc;
  assign sum_prods_fc = prod_fc_0 + prod_fc_1 + prod_fc_2 + prod_fc_3;

  // Only not ready for a command when we have a response.
  assign cmd_ready = ~rsp_valid;

  always @(posedge clk) begin
    if (reset) begin
      rsp_payload_outputs_0 <= 32'b0;
      rsp_valid <= 1'b0;
      InputOffset <= 0;
      FilterOffset <= 0;
    end else if (rsp_valid) begin
      // Waiting to hand off response to CPU.
      rsp_valid <= ~rsp_ready;
    end else if (cmd_valid) begin
      rsp_valid <= 1'b1;
      // Accumulate step:
      case (cmd_payload_function_id[9:3])
        2'b000_0000: begin
          InputOffset <= InputOffset;
          rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods;
        end
        2'd1: begin
          InputOffset <= cmd_payload_inputs_0[15:0];
          FilterOffset <= cmd_payload_inputs_1[15:0];
          rsp_payload_outputs_0 <= 0'b0;
        end
        2'd2: begin
          FilterOffset <= FilterOffset;
          InputOffset <= InputOffset;
          rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods_fc;
        end
        7'd3: begin
          InputOffset <= cmd_payload_inputs_0[15:0];
          FilterOffset <= cmd_payload_inputs_1[15:0];
          rsp_payload_outputs_0 <= 0'b0;
        end
        default: begin
          InputOffset <= InputOffset;
          rsp_payload_outputs_0 <= 0'b0;
        end
      endcase
    end
  end
endmodule