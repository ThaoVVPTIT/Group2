`timescale 1ns / 1ps
/*------------------------------------------------------------------------
 *  Module: conv2_buf
 *  Design : Bo dem dau vao cho lop tich chap thu hai (Conv2)
 *           Dau vao: 6 kenh x 13 x 13 (tu pool1), kernel 3x3.
 *           Vi co 6 kenh dau vao (khac conv1_buf chi co 1 kenh), module
 *           nay dung 6 instance conv_line_buf song song (1 instance/kenh)
 *           - dung y tuong "16 Feature Buffer CHx" trong so do khoi,
 *           thu nho lai cho 6 kenh dau vao cua conv2.
 *           Tat ca 6 kenh nhan CUNG mot valid_in (vi 6 kenh luon di
 *           kem nhau tren cung 1 "tick" du lieu tu maxpool_relu truoc),
 *           nen valid_out_buf cua ca 6 kenh luon dong bo - chi can lay
 *           1 tin hieu dai dien.
 *------------------------------------------------------------------------*/
module conv2_buf
    #(
        parameter WIDTH     = 13,
        parameter HEIGHT    = 13,
        parameter DATA_BITS = 8,
        parameter IN_CH     = 6
    )
    (
        input  wire                        clk,
        input  wire                        rst_n,
        input  wire                        valid_in,
        input  wire [(IN_CH*DATA_BITS)-1:0] data_in_flat,
        output wire [(IN_CH*9*DATA_BITS)-1:0] win_flat,   // IN_CH kenh x 9 tap
        output wire                        valid_out_buf
    );

    wire signed [DATA_BITS-1:0] data_in [0:IN_CH-1];
    genvar gi_in;
    generate
        for (gi_in=0; gi_in<IN_CH; gi_in=gi_in+1) begin : gen_din_unpack
            assign data_in[gi_in] = data_in_flat[gi_in*DATA_BITS +: DATA_BITS];
        end
    endgenerate

    wire signed [DATA_BITS-1:0] win [0:IN_CH*9-1];
    genvar gi_out;
    generate
        for (gi_out=0; gi_out<IN_CH*9; gi_out=gi_out+1) begin : gen_win_pack
            assign win_flat[gi_out*DATA_BITS +: DATA_BITS] = win[gi_out];
        end
    endgenerate

    wire [IN_CH-1:0] valid_out_buf_ch;
    assign valid_out_buf = valid_out_buf_ch[0];

    genvar ch;
    generate
        for (ch = 0; ch < IN_CH; ch = ch + 1) begin : g_ch
            wire [(9*DATA_BITS)-1:0] win_ch_flat;
            wire signed [DATA_BITS-1:0] win_ch [0:8];
            genvar gi_wch;
            for (gi_wch=0; gi_wch<9; gi_wch=gi_wch+1) begin : gen_wch_unpack
                assign win_ch[gi_wch] = win_ch_flat[gi_wch*DATA_BITS +: DATA_BITS];
            end

            conv_line_buf
                #(
                    .WIDTH(WIDTH), .HEIGHT(HEIGHT), .DATA_BITS(DATA_BITS), .FILTER(3)
                )
                u_line_buf
                (
                    .clk(clk), .rst_n(rst_n),
                    .valid_in(valid_in), .data_in(data_in[ch]),
                    .win_flat(win_ch_flat), .valid_out_buf(valid_out_buf_ch[ch])
                );

            assign win[ch*9+0] = win_ch[0];
            assign win[ch*9+1] = win_ch[1];
            assign win[ch*9+2] = win_ch[2];
            assign win[ch*9+3] = win_ch[3];
            assign win[ch*9+4] = win_ch[4];
            assign win[ch*9+5] = win_ch[5];
            assign win[ch*9+6] = win_ch[6];
            assign win[ch*9+7] = win_ch[7];
            assign win[ch*9+8] = win_ch[8];
        end
    endgenerate

endmodule
