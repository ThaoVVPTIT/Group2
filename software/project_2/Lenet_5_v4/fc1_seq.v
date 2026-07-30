`timescale 1ns / 1ps
// =============================================================
// FC1 : Lop ket noi day du 400 -> 120, kem quant + ReLU + clamp
// =============================================================
module fc1_seq (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    output wire        ready,
    output reg         done,

    // Cong ghi vector dau vao (400 phan tu)
    input  wire        in_wr_en,
    input  wire [8:0]  in_wr_addr,
    input  wire signed [7:0] in_wr_data,

    // Cong doc vector dau ra (120 phan tu)
    input  wire [6:0]  out_rd_addr,
    output wire signed [7:0]  out_rd_data
);

    reg signed [7:0]  f1_kernel [0:47999]; // 120 * 400
    reg signed [31:0] f1_bias   [0:119];
    reg signed [31:0] f1_mult   [0:119];
    reg [7:0]         f1_shift  [0:119];

    initial begin
        $readmemh("fc1_kernel.mem",     f1_kernel);
        $readmemh("fc1_bias.mem",       f1_bias);
        $readmemh("fc1_multiplier.mem", f1_mult);
        $readmemh("fc1_shift.mem",      f1_shift);
    end

    reg signed [7:0] p2_out  [0:399];
    reg signed [7:0] fc1_out [0:119];

    always @(posedge clk) begin
        if (in_wr_en) p2_out[in_wr_addr] <= in_wr_data;
    end
    assign out_rd_data = fc1_out[out_rd_addr];

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
                    acc <= f1_bias[c];
                    kc <= 0;
                    state <= MAC;
                end

                MAC: begin
                    acc <= acc + p2_out[kc] * f1_kernel[c*400 + kc];
                    if (kc == 399) state <= QUANT;
                    else kc <= kc + 1;
                end

                QUANT: begin
                    prod = acc * f1_mult[c];
                    if (f1_shift[c] > 0) begin
                        prod = prod + (64'sd1 << (f1_shift[c] - 1));
                        acc = prod >>> f1_shift[c];
                    end else acc = prod;

                    // ReLU: ep gia tri am ve 0 (khop voi golden model
                    // trong train.py: val = np.maximum(val, 0))
                    // KHONG dung acc = acc - 128 nhu cu (sai so voi golden model)
                    if (acc < 0) acc = 0;
                    if (acc > 127) acc = 127; else if (acc < -128) acc = -128;

                    fc1_out[c] <= acc[7:0];

                    if (c == 119) begin
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
