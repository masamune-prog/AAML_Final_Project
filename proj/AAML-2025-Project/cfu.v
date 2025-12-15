`include "RTL/TPU.v"
`include "global_buffer_bram.v"

module Cfu
#(
  parameter ADDR_BITS = 12,
  parameter DATA_BITS = 32,
  parameter C_BITS    = 128
)(
  input               cmd_valid,
  output  reg         cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output  reg         rsp_valid,
  input               rsp_ready,
  output  reg [31:0]  rsp_payload_outputs_0,
  input               reset,
  input               clk
);

  // Control and params
  reg         rst_n;
  reg         in_valid;
  reg [31:0]  K, M, N;

  wire [6:0]  op = cmd_payload_function_id[9:3];
  reg  [31:0] input_offset;
  // TPU interface
  wire        busy;
  wire        tpu_active = busy | in_valid;

  // Global buffer signals
  wire [31:0]       A_data_out, B_data_out;
  wire [C_BITS-1:0] C_data_out;
  wire              A_wr_en, B_wr_en, C_wr_en;
  wire [ADDR_BITS-1:0] A_index, B_index, C_index;
  wire [31:0]          A_data_in, B_data_in;
  wire [C_BITS-1:0]    C_data_in;

  // CPU init path (direct writes/reads)
  reg                 A_wr_en_init, B_wr_en_init, C_wr_en_init;
  reg [ADDR_BITS-1:0] A_index_init, B_index_init, C_index_init;
  reg [31:0]          A_data_in_init, B_data_in_init;
  reg [C_BITS-1:0]    C_data_in_init;

  // Mux TPU vs init path
  wire A_wr_en_mux       = tpu_active ? A_wr_en : A_wr_en_init;
  wire B_wr_en_mux       = tpu_active ? B_wr_en : B_wr_en_init;
  wire C_wr_en_mux       = tpu_active ? C_wr_en : C_wr_en_init;
  wire [ADDR_BITS-1:0] A_index_mux = tpu_active ? A_index : A_index_init;
  wire [ADDR_BITS-1:0] B_index_mux = tpu_active ? B_index : B_index_init;
  wire [ADDR_BITS-1:0] C_index_mux = tpu_active ? C_index : C_index_init;
  wire [31:0]          A_data_in_mux = tpu_active ? A_data_in : A_data_in_init;
  wire [31:0]          B_data_in_mux = tpu_active ? B_data_in : B_data_in_init;
  wire [C_BITS-1:0]    C_data_in_mux = tpu_active ? C_data_in : C_data_in_init;

  // BRAMs (rst_n is unused in this module; tie high)
  global_buffer_bram #(.ADDR_BITS(ADDR_BITS), .DATA_BITS(DATA_BITS)) gbuff_A (
    .clk(clk), .rst_n(1'b1), .ram_en(1'b1),
    .wr_en(A_wr_en_mux), .index(A_index_mux),
    .data_in(A_data_in_mux), .data_out(A_data_out)
  );
  global_buffer_bram #(.ADDR_BITS(ADDR_BITS), .DATA_BITS(DATA_BITS)) gbuff_B (
    .clk(clk), .rst_n(1'b1), .ram_en(1'b1),
    .wr_en(B_wr_en_mux), .index(B_index_mux),
    .data_in(B_data_in_mux), .data_out(B_data_out)
  );
  global_buffer_bram #(.ADDR_BITS(ADDR_BITS), .DATA_BITS(C_BITS)) gbuff_C (
    .clk(clk), .rst_n(1'b1), .ram_en(1'b1),
    .wr_en(C_wr_en_mux), .index(C_index_mux),
    .data_in(C_data_in_mux), .data_out(C_data_out)
  );

  TPU tpu(
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_valid),
    .K(K), .M(M), .N(N),
    .busy(busy),
    .A_wr_en(A_wr_en),
    .A_index(A_index),
    .A_data_in(A_data_in),
    .A_data_out(A_data_out),
    .B_wr_en(B_wr_en),
    .B_index(B_index),
    .B_data_in(B_data_in),
    .B_data_out(B_data_out),
    .C_wr_en(C_wr_en),
    .C_index(C_index),
    .C_data_in(C_data_in),
    .C_data_out(C_data_out),
    .input_offset(input_offset)
  );

  // Command FSM (non-blocking to compute)
  localparam S_IDLE      = 2'd0;
  localparam S_DECODE    = 2'd1;
  localparam S_READ_WAIT = 2'd2;
  localparam S_RESPOND   = 2'd3;

  reg [1:0] state, next_state;

  always @(posedge clk) begin
    if (reset) state <= S_IDLE;
    else       state <= next_state;
  end

  always @* begin
    next_state = state;
    case (state)
      S_IDLE:      if (cmd_valid) next_state = S_DECODE;
      S_DECODE: begin
        //read
        if (op == 7'd9 || op == 7'd11 || (op >= 7'd14 && op <= 7'd17))
          next_state = S_READ_WAIT;
        else
          next_state = S_RESPOND;
      end
      S_READ_WAIT: 
        next_state = S_RESPOND;  // allow BRAM negedge to update data_out
      S_RESPOND:   
        if (rsp_ready) next_state = S_IDLE;
      default:     next_state = S_IDLE;
    endcase
  end

  // Main control/IO
  always @(posedge clk) begin
    if (reset) begin
      rst_n <= 1'b1;
      cmd_ready <= 1'b0;
      rsp_valid <= 1'b0;
      rsp_payload_outputs_0 <= 32'd0;
      in_valid <= 1'b0;
      K <= 0; 
      M <= 0; 
      N <= 0; 
      input_offset <= 0;

      A_wr_en_init <= 0; B_wr_en_init <= 0; C_wr_en_init <= 0;
      A_index_init <= 0; B_index_init <= 0; C_index_init <= 0;
      A_data_in_init <= 0; B_data_in_init <= 0; C_data_in_init <= 0;
    end else begin
      // defaults
      cmd_ready <= 1'b0;
      rsp_valid <= 1'b0;
      in_valid  <= 1'b0;
      A_wr_en_init <= 1'b0;
      B_wr_en_init <= 1'b0;
      C_wr_en_init <= 1'b0;


      case (state)
        S_IDLE: begin
          // hold
        end

        S_DECODE: begin
          cmd_ready <= 1'b1;
          case (op)
            // Reset pulse to TPU domain (active-low)
            7'd1: rst_n <= 1'b0;

            // Set/get K/M/N
            7'd2: K <= cmd_payload_inputs_0;
            7'd3: rsp_payload_outputs_0 <= K;
            7'd4: M <= cmd_payload_inputs_0;
            7'd5: rsp_payload_outputs_0 <= M;
            7'd6: N <= cmd_payload_inputs_0;
            7'd7: rsp_payload_outputs_0 <= N;

            // A write/read
            7'd8: begin
              A_index_init   <= cmd_payload_inputs_0[ADDR_BITS-1:0];
              A_data_in_init <= cmd_payload_inputs_1;
              A_wr_en_init   <= 1'b1;
            end
            7'd9: begin
              A_index_init <= cmd_payload_inputs_0[ADDR_BITS-1:0];
            end

            // B write/read
            7'd10: begin
              B_index_init   <= cmd_payload_inputs_0[ADDR_BITS-1:0];
              B_data_in_init <= cmd_payload_inputs_1;
              B_wr_en_init   <= 1'b1;
            end
            7'd11: begin
              B_index_init <= cmd_payload_inputs_0[ADDR_BITS-1:0];
            end

            // Start compute; report busy
            7'd12: begin
              rst_n <= 1'b1;           // ensure released
              in_valid <= 1'b1;        // pulse start
              rsp_payload_outputs_0 <= busy;
            end

            // Query busy
            7'd13: rsp_payload_outputs_0 <= busy;

            // C read (4x 32-bit slices)
            7'd14, 7'd15, 7'd16, 7'd17: begin
              C_index_init <= cmd_payload_inputs_0[ADDR_BITS-1:0];
            end
            7'd18: begin // Pass Offset
            input_offset <= cmd_payload_inputs_0;
          end
            default: ;
          endcase
        end

        S_READ_WAIT: begin
          case (op)
            7'd9:  rsp_payload_outputs_0 <= A_data_out;
            7'd11: rsp_payload_outputs_0 <= B_data_out;
            7'd14: rsp_payload_outputs_0 <= C_data_out[31:0];
            7'd15: rsp_payload_outputs_0 <= C_data_out[63:32];
            7'd16: rsp_payload_outputs_0 <= C_data_out[95:64];
            7'd17: rsp_payload_outputs_0 <= C_data_out[127:96];
            default: rsp_payload_outputs_0 <= 32'd0;
          endcase
        end

        S_RESPOND: begin
          //set rep_valid to do handshake to signal output ready
          rsp_valid <= 1'b1;
          if (op == 7'd1 && rsp_ready) rst_n <= 1'b1; // release reset after ack
          if (op == 7'd13) rsp_payload_outputs_0 <= busy;
        end
      endcase
    end
  end

endmodule