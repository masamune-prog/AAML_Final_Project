`include "RTL/systolic_array.v"
`include "RTL/dataloader.v"
`include "RTL/controller.v"
module TPU
#(
    parameter ADDR_BITS=15
)(
    clk,
    rst_n,

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
    input_offset
);


input clk;
input rst_n;
input            in_valid;
input [31:0]      K;
input [31:0]      M;
input [31:0]      N;
output  reg      busy;

output           A_wr_en;
output [ADDR_BITS-1:0]    A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output [ADDR_BITS-1:0]    B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output           C_wr_en;
output [ADDR_BITS-1:0]    C_index;
output [127:0]   C_data_in;
input  [127:0]   C_data_out;
input [31:0] input_offset;
reg [31:0] K_reg, M_reg, N_reg;


wire [1:0] state, next_state;
wire [31:0] counter;
wire [31:0] in_left, in_up;
wire [127:0] psum_1;
wire [127:0] psum_2;
wire [127:0] psum_3;
wire [127:0] psum_4;

//load data
parameter IDLE = 2'd0;
parameter READ = 2'd1;
parameter WRITE = 2'd2;
parameter FINISH = 2'd3;

//activate the checking
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 1'b0;
    end else if (in_valid) begin
        busy <= 1'b1;
    end else if (next_state == FINISH) begin
        busy <= 1'b0;
    end
end



always @(posedge clk) begin
    if(K > 0) begin 
        K_reg <= K;
        M_reg <= M;
        N_reg <= N;
    end
end

data_loader data_loader_a (
    .clk(clk),
    .rst_n(rst_n),
    .state(state),
    .input_data(A_data_out),
    .K_reg(K_reg), 
    .counter(counter), 
    .output_wire(in_left)
);
data_loader data_loader_b (
    .clk(clk),
    .rst_n(rst_n),
    .state(state),
    .input_data(B_data_out),
    .K_reg(K_reg), 
    .counter(counter), 
    .output_wire(in_up)
);
systolic_array sys_array (
    .clk(clk),
    .rst_n(rst_n),
    .state(state),
    .up_in(in_up),
    .left_in(in_left),
    .psum_1(psum_1),
    .psum_2(psum_2),
    .psum_3(psum_3),
    .psum_4(psum_4),
    .input_offset(input_offset)
);
controller ctrl(
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_valid),
    .busy(busy),
    .K_reg(K_reg),
    .M_reg(M_reg),
    .N_reg(N_reg),
    .psum_1(psum_1),
    .psum_2(psum_2),
    .psum_3(psum_3),
    .psum_4(psum_4),
    .state_wire(state),
    .next_state_wire(next_state),
    .A_wr_en(A_wr_en),
    .B_wr_en(B_wr_en),
    .C_wr_en(C_wr_en),
    .A_data_in(A_data_in),
    .B_data_in(B_data_in),
    .C_data_in(C_data_in),
    .A_index(A_index),
    .B_index(B_index),
    .C_index(C_index),
    .counter_wire(counter)
);



endmodule