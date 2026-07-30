`timescale 1ns / 1ps
/*------------------------------------------------------------------------
 *  Module: conv2_layer
 *  Design : Lop tich chap thu hai cho CNN EMNIST 3x3 (6 -> 16 kenh)
 *------------------------------------------------------------------------*/
module conv2_layer
    (
        input  wire        clk,
        input  wire        rst_n,
        input  wire        valid_in,
        input  wire [(6*8)-1:0] data_in_flat,
        output wire [(16*8)-1:0] conv2_out_flat,
        output wire        valid_out_conv2
    );

    wire [(54*8)-1:0] win_flat;
    wire valid_out_buf;

    conv2_buf u_conv2_buf
        (
            .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .data_in_flat(data_in_flat),
        .win_flat(win_flat), .valid_out_buf(valid_out_buf)
    );

    conv2_calc #(.DATA_BITS(8)) u_conv2_calc (
        .clk(clk), .rst_n(rst_n), .valid_out_buf(valid_out_buf),
        .win_flat(win_flat),
        .conv_out_flat(conv2_out_flat), .valid_out_calc(valid_out_conv2)
        );

endmodule
