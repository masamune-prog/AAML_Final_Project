`timescale 1ns/1ps
`include "RTL/systolic_array.v"

module systolic_array_tb;

    // Parameters
    parameter SIZE = 4;
    parameter DATA_WIDTH = 8;
    parameter BUS_WIDTH  = 32;

    // Signals
    reg clk, rst_n;
    reg [1:0] state;
    reg [BUS_WIDTH-1:0] up_in;
    reg [BUS_WIDTH-1:0] left_in;

    wire [127:0] psum_1;
    wire [127:0] psum_2;
    wire [127:0] psum_3;
    wire [127:0] psum_4;

    // Instantiate DUT
    systolic_array #(
        .SIZE(SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .BUS_WIDTH(BUS_WIDTH)
    ) DUT (
        .rst_n(rst_n),
        .clk(clk),
        .up_in(up_in),
        .left_in(left_in),
        .state(state),
        .psum_1(psum_1),
        .psum_2(psum_2),
        .psum_3(psum_3),
        .psum_4(psum_4)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;

    // Test stimulus
    initial begin
        $dumpfile("systolic_array_tb.vcd");
        $dumpvars(0, systolic_array_tb);

        // Reset
        rst_n = 0;
        state = 2'b00;
        up_in = 0;
        left_in = 0;
        #20;

        rst_n = 1;
        state = 2'b01; // active state for operation

        // ----------------------------------------------------------
        //  Test case: 2x2 Matrix multiplication
        //
        //  A = [ 1 2 ]
        //      [ 3 4 ]
        //
        //  B = [ 5 6 ]
        //      [ 7 8 ]
        //
        //  Expected C = A × B =
        //      [ (1*5 + 2*7)   (1*6 + 2*8) ] = [19 22]
        //      [ (3*5 + 4*7)   (3*6 + 4*8) ] = [43 50]
        // ----------------------------------------------------------

        // We only use top-left 2x2 region
        // Each BUS_WIDTH=32 carries 4x8-bit = 4 elements per row/column

        // Cycle 1: send first column of A on left_in, first row of B on up_in
        // left_in: {A[1,0], A[0,0]} = {3,1}
        // up_in  : {B[0,1], B[0,0]} = {6,5}
        left_in = {8'd0, 8'd0, 8'd3, 8'd1};
        up_in   = {8'd0, 8'd0, 8'd6, 8'd5};
        #10;

        // Cycle 2: send next column/row
        // left_in: {A[1,1], A[0,1]} = {4,2}
        // up_in  : {B[1,1], B[1,0]} = {8,7}
        left_in = {8'd0, 8'd0, 8'd4, 8'd2};
        up_in   = {8'd0, 8'd0, 8'd8, 8'd7};
        #10;

        // Keep steady for systolic flow to propagate
        left_in = 0;
        up_in   = 0;
        #80;

        $display("====================================");
        $display("Final PSUM outputs:");
        $display("psum_1 = %h", psum_1);
        $display("psum_2 = %h", psum_2);
        $display("psum_3 = %h", psum_3);
        $display("psum_4 = %h", psum_4);
        $display("====================================");

        #20;
        $finish;
    end

endmodule
