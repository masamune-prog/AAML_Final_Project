`include "RTL/PE.v"
module systolic_array #(
    parameter SIZE = 4,
    parameter DATA_WIDTH = 8,
    parameter BUS_WIDTH = 32
)(
    rst_n,
    clk,
    up_in,
    left_in,
    state,
    psum_1,
    psum_2,
    psum_3,
    psum_4,
    input_offset
);



input clk,rst_n;
input [1:0] state;
input [BUS_WIDTH-1:0] left_in;
input [BUS_WIDTH-1:0] up_in;
input [31:0] input_offset;
//output a 512 bit cause each PE output 32bit and we have 16 PE in total since we concat the ouputs from each pe

//wire used as we just extracting data from within the PE
output wire [127:0] psum_1;
output wire [127:0] psum_2;
output wire [127:0] psum_3;
output wire [127:0] psum_4;
//generate the wires within the 4x4 grid we need 48 wires in total
//We make 16 wires here
wire [DATA_WIDTH-1:0] out_right [0:SIZE-1][0:SIZE-1];
//another 16
wire [DATA_WIDTH-1:0] out_down [0:SIZE-1][0:SIZE-1];
//16 wires to extract the values from within the PE
wire [31:0] psum [0:SIZE*SIZE-1];

//generate the 4x4

genvar i, j;
generate
    for(i=0;i<SIZE;i = i+1) begin : ROW
        for (j=0;j<SIZE;j = j+1) begin : COL
            localparam IDX = i * SIZE + j;
            //16 output and input  wires
            wire [DATA_WIDTH-1:0] in_left;
            wire [DATA_WIDTH-1:0] in_up;
            // Hold PE in reset during IDLE so accumulators clear between runs
            wire pe_rst_n = (state == 2'b00) ? 1'b0 : 1'b1;
            // Connect left
            if (j == 0) begin
                //get from input
                assign in_left = left_in[(BUS_WIDTH - DATA_WIDTH * i - 1) -: DATA_WIDTH];
            end
            else begin
                //push output to the right
                assign in_left = out_right[i][j-1];
            end

            // Connect up input
            if (i == 0) begin
                assign in_up = up_in[(BUS_WIDTH - DATA_WIDTH * j - 1) -: DATA_WIDTH];
            end
            else begin
                //push output down
                assign in_up = out_down[i-1][j];
            end
            PE pe(
                .clk(clk),
                .rst_n(pe_rst_n),
                .left_in(in_left),
                .up_in(in_up),
                .right_out(out_right[i][j]),
                .down_out(out_down[i][j]),
                .result_out(psum[IDX]),
                .input_offset(input_offset)
            );
        end
    end

endgenerate
assign psum_1 = (state == 2'd2) ? {psum[0], psum[1], psum[2], psum[3]} : 128'b0;
assign psum_2 = (state == 2'd2) ? {psum[4], psum[5], psum[6], psum[7]}: 128'b0;
assign psum_3 = (state == 2'd2) ? {psum[8], psum[9], psum[10], psum[11]}: 128'b0;
assign psum_4 = (state == 2'd2) ? {psum[12], psum[13], psum[14], psum[15]}: 128'b0;


endmodule