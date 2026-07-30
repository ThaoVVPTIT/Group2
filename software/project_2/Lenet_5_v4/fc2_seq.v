`timescale 1ns / 1ps
// =============================================================
// FC2 : Lop ket noi day du 120 -> 84, kem quant + ReLU + clamp
// =============================================================
module fc2_seq (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    output wire        ready,
    output reg         done,

    input  wire        in_wr_en,
    input  wire [6:0]  in_wr_addr,
    input  wire signed [7:0] in_wr_data,

    input  wire [6:0]  out_rd_addr,
    output wire signed [7:0]  out_rd_data
);

    reg signed [7:0]  f2_kernel [0:10079]; // 84 * 120
    reg signed [31:0] f2_bias   [0:83];
    reg signed [31:0] f2_mult   [0:83];
    reg [7:0]         f2_shift  [0:83];

    initial begin
        $readmemh("fc2_kernel.mem",     f2_kernel);
        $readmemh("fc2_bias.mem",       f2_bias);
        $readmemh("fc2_multiplier.mem", f2_mult);
        $readmemh("fc2_shift.mem",      f2_shift);
    end

    reg signed [7:0] fc1_out [0:119];
    reg signed [7:0] fc2_out [0:83];

    always @(posedge clk) begin
        if (in_wr_en) fc1_out[in_wr_addr] <= in_wr_data;
    end
    assign out_rd_data = fc2_out[out_rd_addr];

    localparam IDLE = 0, INIT = 1, MAC = 2, QUANT = 3;
    reg [2:0] state;

    reg [15:0] c, kc;
    reg signed [31:0] acc;
    reg signed [63:0] prod;

    assign ready = (state == IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done  <= 0;
        end else begin
            done <= 0;
            case (state)
                IDLE: begin
                    if (start) begin
                        c <= 0;
                        state <= INIT;
                    end
                end

                INIT: begin
                    acc <= f2_bias[c];
                    kc <= 0;
                    state <= MAC;
                end

                MAC: begin
                    acc <= acc + fc1_out[kc] * f2_kernel[c*120 + kc];
                    if (kc == 119) state <= QUANT;
                    else kc <= kc + 1;
                end

                QUANT: begin
                    prod = acc * f2_mult[c];
                    if (f2_shift[c] > 0) begin
                        prod = prod + (64'sd1 << (f2_shift[c] - 1));
                        acc = prod >>> f2_shift[c];
                    end else acc = prod;

                    // ReLU: ep gia tri am ve 0 (khop voi golden model)
                    if (acc < 0) acc = 0;
                    if (acc > 127) acc = 127; else if (acc < -128) acc = -128;
                    fc2_out[c] <= acc[7:0];

                    if (c == 83) begin
                        done  <= 1;
                        state <= IDLE;
                    end else begin
                        c <= c + 1;
                        state <= INIT;
                    end
                end
            endcase
        end
    end
endmodule
