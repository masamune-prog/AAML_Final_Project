//FSM to handle the stages where we invoke the
module controller#(
    parameter ADDR_BITS=12
)(
    clk,
    rst_n,
    in_valid,
    busy,
    K_reg,
    M_reg,
    N_reg,
    psum_1,
    psum_2,
    psum_3,
    psum_4,
    state_wire,
    next_state_wire,
    A_wr_en,
    B_wr_en,
    C_wr_en,
    A_data_in,
    B_data_in,
    C_data_in,
    A_index,
    B_index,
    C_index,
    counter_wire
);

input clk,rst_n,in_valid,busy;
//keep the size
input [31:0] K_reg,M_reg,N_reg;

input [127:0] psum_1;
input [127:0] psum_2;
input [127:0] psum_3;
input [127:0] psum_4;

output wire [1:0] state_wire,next_state_wire;
output A_wr_en;
output B_wr_en;
output C_wr_en;
output [31:0] A_data_in;
output [31:0] B_data_in;
output [127:0] C_data_in;
output [ADDR_BITS-1:0] A_index;
output [ADDR_BITS-1:0] B_index;
output [ADDR_BITS-1:0] C_index;
output wire [31:0] counter_wire;

reg [2:0] out_cycle;

//state registers for FSM

reg [1:0] state,next_state;

//counters track which part of a,b or c we are reading/writing
reg [7:0] counter_a;
reg [7:0] counter_b;
reg [31:0] counter;
reg [31:0] counter_out; 
//buffer_index

reg [15:0] idx_a;
reg [15:0] idx_b;
reg [15:0] idx_c;


reg [7:0] a_offset; //how many times window needs to be slided
reg [7:0] b_offset;

//We write a FSM to handle the pipelining of the data
parameter IDLE = 2'd0;
parameter READ = 2'd1;
parameter WRITE = 2'd2;
parameter FINISH = 2'd3;

assign state_wire = state;
assign next_state_wire = next_state;
assign counter_wire = counter;
assign A_index = (state == READ) ? idx_a : 0;
assign B_index = (state == READ) ? idx_b : 0;
assign C_index = idx_c;

//handle reset and fsm transtion
always @(posedge clk or negedge rst_n) begin
        if(!rst_n) 
            state <= IDLE;
        else 
            state <= next_state;
    end

//FSM definition

always @(*) begin
    case(state)
    IDLE:
    begin
        //$display("IDLE");
        if(in_valid || busy)begin
            next_state = READ;
        end
        else begin
            next_state = IDLE;
        end
    end
    READ:
    begin
        //$display("READ");
        //latency to complete is 6 cycles, 
        //(4 - 1) * 2 = 6 extra cycles to allow the flushing and Propagation
        // Stay in the READ state until we’ve fed in all K columns of A and rows of B, and allowed 6 extra cycles for the pipeline to fill and flush through the 4×4 array.
        if(counter<=(K_reg+6))begin
            next_state = READ;
        end
        else begin
            next_state = WRITE;
        end
    end
    WRITE:
    begin
        //$display("WRITE");
        // Continue writing until we've emitted all rows for this 4x4 tile
        if(counter_out < out_cycle) begin
            next_state = WRITE;
        end
        // Only finish after completing the last column tile AND the last row tile
        else if (counter_b == b_offset - 1 && counter_a == a_offset) begin
            next_state = FINISH;
        end
        // Otherwise, move back to IDLE to set up the next tile
        else begin
            next_state = IDLE;
        end
    end
    FINISH:
    begin
        //$display("FINISH");
        next_state = IDLE;
    end
    default:
        next_state = IDLE;

    endcase
end
assign A_wr_en = 0;
assign B_wr_en = 0;
assign C_wr_en = (next_state == WRITE) ? 1 : 0;
// block offset
//part 4
always @(*) begin
    a_offset = ((M_reg+3)/4); //Trick for ceil(M/4)
    b_offset = ((N_reg+3)/4); // $ceil(N/4)
end
 // out_cycle
//calculate how many cycles to write
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) out_cycle <= 0;
    else if(state == READ && next_state == WRITE)
        out_cycle <= (counter_a == (a_offset - 1) && M_reg[1:0] != 2'b00 ) ? M_reg[1:0] : 4; 
    else
        out_cycle <= out_cycle;
end
//increase counter during READ
always @(posedge clk or negedge rst_n) begin
        if(!rst_n) counter <= 0;
        else begin
            if(state == READ)
                counter <= counter + 1;
            else
                counter <= 0;
        end
    end
//a counter is for moving to next block of 4x4 in A
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) counter_a <= 0;
    else begin
        //check if we writing and t
        if(state == WRITE && counter_out == out_cycle-1)begin
            //have not reach the end of the input
            if(counter_a < a_offset)begin
                counter_a <= counter_a + 1;
            end
            else
                counter_a<=1;  
        end
        //reset when finish
        else if(state == FINISH)begin
            counter_a <= 0;
        end
    end
end
//compute when b is done, when a is finished
always @(posedge clk or negedge rst_n) begin
        if(!rst_n) counter_b <= 0;
        else begin
            if(next_state == IDLE && counter_a == a_offset && busy)
                counter_b <= counter_b + 1;
            else if(state == FINISH)
                counter_b <= 0;
        end
    end
//when we have output we will add counter_out to write, we need to also ensure we output the right number of cycles based off the dim of the output
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)begin
        counter_out <=0;
    end
    else if(state == WRITE) begin
        counter_out <= counter_out+1;
    end
    else
        counter_out<=0;
end


// buffer part we want to write to buffer c and read from buffer a and b
// index arg from the system sram interface

//retrive from buffer a
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) idx_a <= 15'd0;
        else begin
        if(state == WRITE) begin
            if(K_reg == 1)
                idx_a <= 1;
            else if(next_state==IDLE) begin
                idx_a <= idx_a + 15'd1;
            end
            else
              idx_a <= idx_a;
        end
        else if(state == IDLE && counter_a == a_offset)begin
            idx_a <= 0;
        end
        else if(state == FINISH)begin
            idx_a <= 15'd0;
        end
        else if(state == READ) begin
            if(counter < K_reg - 1) 
            idx_a <= idx_a + 15'd1;
        end
        else begin
            idx_a <= idx_a;
        end
        end
    end
  
    // idx_b
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) idx_b <= 0;
        else begin
            //jump to right block
            if(state == IDLE && busy) begin
                idx_b <= K_reg * counter_b; 
            end
            else if(state == FINISH)begin
                idx_b <= 15'd0;
            end
            else if(state == READ) begin
                // 0 indexed
                if(counter < K_reg ) 
                    idx_b <= idx_b + 1;
            end
        end
    end
//idx_c 

always @(posedge clk or negedge rst_n)begin
    if(!rst_n) begin
        idx_c <= 0;
    end
    else begin
        if(state == FINISH)begin
            idx_c <= 0;
        end
        else if(state == WRITE && next_state == WRITE)begin
            //in between writing blks, increase c
            idx_c <= idx_c + 1;
        end
        else
            idx_c <= idx_c;
    end

end

assign A_data_in = 0;
assign B_data_in = 0;
//we output each row into buffer c
assign C_data_in = (!rst_n) ? 0 :
                (counter_out == 2'd0) ? psum_1 :
                (counter_out == 2'd1) ? psum_2 :
                (counter_out == 2'd2) ? psum_3 :
                (counter_out == 2'd3) ? psum_4 : 0;


endmodule