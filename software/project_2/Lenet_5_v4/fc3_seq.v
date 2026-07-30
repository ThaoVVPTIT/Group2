`timescale 1ns / 1ps
// =============================================================
// FC3 : Lop ket noi day du 84 -> 47 (lop logit dau ra)
//
// Day la lop cuoi cung (output layer) cua CNN, ket qua la 47
// gia tri logit (raw score) duoc dua vao argmax47 de tim lop
// co score cao nhat.
//
// KHONG QUANTIZE (khong co fc3_multiplier.mem, fc3_shift.mem):
// vi chi can SO SANH (argmax), khong can quy ve int8.
// Output la int32 (bias + MAC), giu nguyen do rong de khong
// mat thong tin thu tu giua cac lop co logit gan nhau.
// =============================================================
module fc3_seq (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    output wire        ready,
    output reg         done,

    input  wire        in_wr_en,
    input  wire [6:0]  in_wr_addr,
    input  wire signed [7:0] in_wr_data,

    // Dau ra la logit tho (32-bit, KHONG quantize, KHONG clamp)
    input  wire [5:0]  out_rd_addr,
    output wire signed [31:0] out_rd_data
);

    reg signed [7:0]  f3_kernel [0:3947]; // 47 * 84
    reg signed [31:0] f3_bias   [0:46];

    initial begin
        $readmemh("fc3_kernel.mem", f3_kernel);
        $readmemh("fc3_bias.mem",   f3_bias);
    end

    reg signed [7:0]  fc2_out [0:83];
    reg signed [31:0] fc3_out [0:46];

    always @(posedge clk) begin
        if (in_wr_en) fc2_out[in_wr_addr] <= in_wr_data;
    end
    assign out_rd_data = fc3_out[out_rd_addr];

    localparam IDLE = 0, INIT = 1, MAC = 2, STORE = 3;
    reg [2:0] state;

    reg [15:0] c, kc;
    reg signed [31:0] acc;

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
                    acc <= f3_bias[c];
                    kc <= 0;
                    state <= MAC;
                end

                MAC: begin
                    acc <= acc + fc2_out[kc] * f3_kernel[c*84 + kc];
                    if (kc == 83) state <= STORE;
                    else kc <= kc + 1;
                end

                STORE: begin
                    // Ghi truc tiep acc (bias + MAC) vao output, KHONG quantize
                    fc3_out[c] <= acc;

                    if (c == 46) begin
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

