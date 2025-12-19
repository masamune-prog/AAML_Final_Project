module PE(
    rst_n,
    clk,
    up_in,
    left_in,
    right_out,
    down_out,
    result_out,
    input_offset
);

input clk;
input rst_n;
input [31:0] input_offset;
input signed [7:0] up_in, left_in;
output reg signed [7:0] right_out, down_out;
output signed [31:0] result_out;

reg signed [31:0] ans;

// Extend left_in to 9 bits and add input_offset
wire signed [8:0] left_in_ext = $signed(left_in);
wire signed [8:0] left_in_offset = left_in_ext + $signed(input_offset[8:0]);
wire signed [8:0] up_in_ext = $signed(up_in);

// Compute product with offset applied to left_in
wire signed [31:0] product = left_in_offset * up_in_ext;

assign result_out = ans;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        ans <= 0;
        right_out <= 0;
        down_out <= 0;
    end
    else begin
        ans <= ans + product;
        right_out <= left_in;
        down_out <= up_in;
    end
end



endmodule