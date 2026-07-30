`timescale 1ns / 1ps
/*------------------------------------------------------------------------
 *  Module: conv_calc
 *  Vai tro: Loi tinh tich chap dung chung cho conv1_calc.v / conv2_calc.v.
 *           Nhan IN_CH*9 tap tu bo dem cua so truot (conv_line_buf),
 *           tinh IN_CH*9 phep MAC cho tung kenh ra, roi luong tu hoa
 *           bang co che mult/shift + bias + clamp int8 (GIU NGUYEN dung
 *           cong thuc dang dung trong conv1.v/conv2.v hien tai cua ban -
 *           KHONG lam ReLU o day, ReLU duoc gop vao maxpool_relu.v phia
 *           sau, dung nhu ten module "Max Pooling & ReLU" trong so do
 *           khoi CNN Accelerator ban ve).
 *
 *           Pipeline 2 tang (khac voi phong cach "chuoi cong tuan tu
 *           tung so hang 1 gia tri/chu ky" cua conv1_calc.v goc trong
 *           project 20260614_RunOnChip): vi FILTER=3 chi co 9 (hoac 54)
 *           so hang, cong don trong 1 chu ky bang adder-tree to hop la
 *           du nhanh, khong can chia nho lam nhieu chu ky nhu ban goc
 *           (ban goc dung FILTER=5 => 25 so hang, chia lam 4 chu ky).
 *           Day la don gian hoa co chu dich, giu dung 2 buoc chinh
 *           (MAC roi Quant) nhu kien truc goc, chi khac so chu ky.
 *------------------------------------------------------------------------*/
module conv_calc
    #(
        parameter IN_CH      = 1,
        parameter OUT_CH     = 6,
        parameter DATA_BITS  = 8,
        parameter WEIGHT_FILE = "conv1_kernel.mem",
        parameter BIAS_FILE   = "conv1_bias.mem",
        parameter MULT_FILE   = "conv1_multiplier.mem",
        parameter SHIFT_FILE  = "conv1_shift.mem"
    )
    (
        input  wire clk,
        input  wire rst_n,
        input  wire valid_in,
        input  wire [(IN_CH*9*DATA_BITS)-1:0] win_flat,
        output wire [(OUT_CH*DATA_BITS)-1:0] conv_out_flat,
        output reg  valid_out
    );

    wire signed [DATA_BITS-1:0] win [0:IN_CH*9-1];
    genvar gi_win;
    generate
        for (gi_win = 0; gi_win < IN_CH*9; gi_win = gi_win + 1) begin : gen_win_unpack
            assign win[gi_win] = win_flat[gi_win*DATA_BITS +: DATA_BITS];
        end
    endgenerate

    reg signed [DATA_BITS-1:0] conv_out [0:OUT_CH-1];
    genvar gi_out;
    generate
        for (gi_out = 0; gi_out < OUT_CH; gi_out = gi_out + 1) begin : gen_out_pack
            assign conv_out_flat[gi_out*DATA_BITS +: DATA_BITS] = conv_out[gi_out];
        end
    endgenerate

    localparam TAPS = IN_CH*9;

    // ---- ROM trong so (dung dinh dang mult/shift giong lenet5_top.v) ----
    reg signed [DATA_BITS-1:0] kernel [0:OUT_CH-1][0:TAPS-1];
    reg signed [31:0]          bias   [0:OUT_CH-1];
    reg signed [31:0]          mult   [0:OUT_CH-1];
    reg [7:0]                  shift  [0:OUT_CH-1];

    reg signed [DATA_BITS-1:0] kernel_flat [0:OUT_CH*TAPS-1];
    integer oc, k;

    initial begin
        $readmemh(WEIGHT_FILE, kernel_flat);
        $readmemh(BIAS_FILE,   bias);
        $readmemh(MULT_FILE,   mult);
        $readmemh(SHIFT_FILE,  shift);
        for (oc = 0; oc < OUT_CH; oc = oc + 1)
            for (k = 0; k < TAPS; k = k + 1)
                kernel[oc][k] = kernel_flat[oc*TAPS + k];
    end

    // ---- Tang 1: MAC (cong don TAPS phep nhan + bias) ----
    reg signed [31:0] acc [0:OUT_CH-1];
    reg               valid_a;

    // QUAN TRONG: moi khoi always chay song song (cung nhay theo posedge clk)
    // PHAI co bien vong lap RIENG - khong duoc dung chung 1 "integer i" cho
    // nhieu always block, neu khong se xay ra race condition (2 khoi cung
    // ghi/doc chung 1 bien dieu khien vong lap, gay sai ket qua ngay ca khi
    // logic tinh toan hoan toan dung). Day chinh la loi da gap va sua trong
    // testbench truoc do - can luu y tuong tu trong moi module RTL moi.
    integer i_mac;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_a <= 0;
            for (i_mac = 0; i_mac < OUT_CH; i_mac = i_mac + 1) acc[i_mac] <= 0;
        end else begin
            valid_a <= valid_in;
            if (valid_in) begin
                for (i_mac = 0; i_mac < OUT_CH; i_mac = i_mac + 1) begin : mac_ch
                    integer t;
                    reg signed [31:0] sum;
                    sum = bias[i_mac];
                    for (t = 0; t < TAPS; t = t + 1)
                        sum = sum + win[t] * kernel[i_mac][t];
                    acc[i_mac] <= sum;
                end
            end
        end
    end

    // ---- Tang 2: Quant (mult/shift + clamp int8, KHONG ReLU) ----
    reg signed [31:0] q;
    reg signed [63:0] prod;
    integer i_q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
            for (i_q = 0; i_q < OUT_CH; i_q = i_q + 1) conv_out[i_q] <= 0;
        end else begin
            valid_out <= valid_a;
            if (valid_a) begin
                for (i_q = 0; i_q < OUT_CH; i_q = i_q + 1) begin : quant_ch
                    prod = acc[i_q] * mult[i_q];
                    if (shift[i_q] > 0) begin
                        prod = prod + (64'sd1 << (shift[i_q]-1));
                        q = prod >>> shift[i_q];
                    end else q = prod;

                    if (q > 127) q = 127;
                    else if (q < -128) q = -128;

                    // ReLU: ep gia tri am ve 0 (khop voi golden model
                    // trong train.py: val = np.maximum(val, 0))
                    if (q < 0) q = 0;

                    conv_out[i_q] <= q[7:0];
                end
            end
        end
    end

endmodule
