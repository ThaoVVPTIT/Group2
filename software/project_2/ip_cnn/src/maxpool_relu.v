`timescale 1ns / 1ps
/*------------------------------------------------------------------------
 *  Module: maxpool_relu
 *  Design : Max Pooling 2x2 stride 2 + ReLU, gop chung lam 1 khoi (dung
 *           ten "Max Pooling & ReLU" trong so do khoi CNN Accelerator).
 *
 *           QUAN TRONG - khac voi ban dau: module nay nhan them tham so
 *           WIDTH/HEIGHT (kich thuoc THAT cua feature map dau vao) thay
 *           vi chi dung "flag" toggle don gian nhu maxpool_relu.v goc.
 *           Ly do: conv2 cua mang EMNIST 3x3 xuat ra feature map 11x11
 *           (SO LE, khong chia het cho 2!), nen phai CAT BO hang/cot
 *           cuoi cung (floor-division 11/2=5) truoc khi pooling, giong
 *           dung cach pool2.v (kien truc tuan tu cu) da lam. Ban goc
 *           trong project 20260614_RunOnChip khong gap truong hop nay
 *           vi kich thuoc feature map cua ho luon la so chan.
 *
 *           Vi tri (row,col) duoc dem tuong minh (khong dung flag/state
 *           toggle mo ho nhu ban goc); pha tren/duoi va trai/phai suy
 *           truc tiep tu bit thap nhat cua row/col (row[0], col[0]).
 *------------------------------------------------------------------------*/
module maxpool_relu
    #(
        parameter CH        = 6,
        parameter DATA_BITS = 8,
        parameter WIDTH     = 26,   // chieu rong THAT cua feature map dau vao
        parameter HEIGHT    = 26    // chieu cao THAT cua feature map dau vao
    )
    (
        input  wire clk,
        input  wire rst_n,
        input  wire valid_in,
        input  wire [(CH*DATA_BITS)-1:0] conv_out_flat,
        output wire [(CH*DATA_BITS)-1:0] max_value_flat,
        output reg  valid_out_relu
    );

    wire signed [DATA_BITS-1:0] conv_out [0:CH-1];
    genvar gi_in;
    generate
        for (gi_in = 0; gi_in < CH; gi_in = gi_in + 1) begin : gen_conv_unpack
            assign conv_out[gi_in] = conv_out_flat[gi_in*DATA_BITS +: DATA_BITS];
        end
    endgenerate

    reg signed [DATA_BITS-1:0] max_value [0:CH-1];
    genvar gi_out;
    generate
        for (gi_out = 0; gi_out < CH; gi_out = gi_out + 1) begin : gen_max_pack
            assign max_value_flat[gi_out*DATA_BITS +: DATA_BITS] = max_value[gi_out];
        end
    endgenerate

    localparam OUT_W = WIDTH  / 2;   // floor - bo cot cuoi neu WIDTH le
    localparam OUT_H = HEIGHT / 2;   // floor - bo hang cuoi neu HEIGHT le

    reg signed [DATA_BITS-1:0] buffer [0:CH-1][0:OUT_W-1];

    integer col, row;    // vi tri hien tai trong feature map dau vao (0-index)
    integer out_col;     // cot pooling tuong ung (col>>1)

    wire active_col = (col < OUT_W*2);
    wire active_row = (row < OUT_H*2);
    wire col_phase  = col[0];   // 0: cot trai cua cap, 1: cot phai cua cap
    wire row_phase  = row[0];   // 0: hang tren cua cap, 1: hang duoi cua cap

    integer ch, p;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ch = 0; ch < CH; ch = ch + 1) begin : rst_ch
                for (p = 0; p < OUT_W; p = p + 1) buffer[ch][p] <= 0;
                max_value[ch] <= 0;
            end
            col <= 0;
            row <= 0;
            valid_out_relu <= 0;
        end else begin
            valid_out_relu <= 0;

            if (valid_in) begin
                out_col = col >> 1;

                if (active_row && active_col) begin
                    if (row_phase == 1'b0) begin
                        // hang tren cua cap
                        if (col_phase == 1'b0) begin
                            for (ch = 0; ch < CH; ch = ch + 1)
                                buffer[ch][out_col] <= conv_out[ch];
                        end else begin
                            for (ch = 0; ch < CH; ch = ch + 1)
                                if (conv_out[ch] > buffer[ch][out_col])
                                    buffer[ch][out_col] <= conv_out[ch];
                        end
                    end else begin
                        // hang duoi cua cap
                        if (col_phase == 1'b0) begin
                            for (ch = 0; ch < CH; ch = ch + 1)
                                if (conv_out[ch] > buffer[ch][out_col])
                                    buffer[ch][out_col] <= conv_out[ch];
                        end else begin
                            // gia tri cuoi cung cua cua so 2x2 -> xuat (KHONG ep
                            // ve 0 nua: vi kich hoat dung quy uoc zero_point=-128
                            // (chuan TFLite asymmetric int8), gia tri thuc = 0 da
                            // duoc bieu dien boi -128, va buoc clamp [-128,127]
                            // o conv_calc.v da tu dong lam ReLU roi. Neu ep ve 0
                            // o day se SAI vi "0" theo nghia bit khong con la
                            // "gia tri thuc = 0" nua trong quy uoc nay.
                            valid_out_relu <= 1'b1;
                            for (ch = 0; ch < CH; ch = ch + 1) begin : out_ch
                                max_value[ch] <= (conv_out[ch] > buffer[ch][out_col]) ? conv_out[ch] : buffer[ch][out_col];
                            end
                        end
                    end
                end
                // (neu !active_row hoac !active_col: pixel thuoc hang/cot
                //  du ra do WIDTH/HEIGHT le -> bo qua, khong dung den)

                // cap nhat vi tri
                if (col == WIDTH-1) begin
                    col <= 0;
                    if (row == HEIGHT-1) row <= 0;
                    else                 row <= row + 1;
                end else col <= col + 1;
            end
        end
    end

endmodule
