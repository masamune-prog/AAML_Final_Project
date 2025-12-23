//in this module we handle the staggering of the input
module data_loader(
    clk,
    rst_n,
    state,
    input_data,
    K_reg, 
    counter, 
    output_wire
);


input clk,rst_n;
input [1:0] state;
input [31:0] input_data;
//reg to keep dim since they only avail in 1 cycle
//need reg to see number of cycles we need to wait for systolic array to complete
input [31:0] K_reg;
input [31:0] counter;

output wire [31:0] output_wire;

localparam IDLE = 2'b0;
localparam READ = 2'b1;

reg [31:0] out;
reg [7:0] temp_out1;
reg [15:0] temp_out2;
reg [23:0] temp_out3;
reg [31:0] temp_out4;

// Drive the declared output; previously referenced an undeclared net `out_wire`.
assign output_wire = (state == READ) ? out : 32'b0;

//handle the reset and idle case

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out <= 32'd0;
        temp_out1 <= 8'd0;
        temp_out2 <= 16'd0;
        temp_out3 <= 24'd0;
        temp_out4 <= 32'd0;
    end else if (state == IDLE) begin
        out <= 32'd0;
        temp_out1 <= 8'd0;
        temp_out2 <= 16'd0;
        temp_out3 <= 24'd0;
        temp_out4 <= 32'd0;
    end else if (state == READ) begin
        if (counter < K_reg) begin
            temp_out1 <= input_data[31:24];
            temp_out2 <= {temp_out2[7:0],  input_data[23:16]};
            temp_out3 <= {temp_out3[15:0], input_data[15:8]};
            temp_out4 <= {temp_out4[23:0], input_data[7:0]};
            out <= {input_data[31:24], temp_out2[7:0], temp_out3[15:8], temp_out4[23:16]};
        end else begin
            temp_out1 <= temp_out1 << 8;
            temp_out2 <= temp_out2 << 8;
            temp_out3 <= temp_out3 << 8;
            temp_out4 <= temp_out4 << 8;
            out <= {8'b0, temp_out2[7:0], temp_out3[15:8], temp_out4[23:16]};
        end
    end
    // else: hold values
end


endmodule