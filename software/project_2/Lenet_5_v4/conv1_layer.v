`timescale 1ns / 1ps
/*------------------------------------------------------------------------
 *  Module: conv1_layer
 *  Design : Lop tich chap thu nhat cho CNN EMNIST 3x3 (1 -> 6 kenh)
 *------------------------------------------------------------------------*/
module conv1_layer
    (
        input  wire        clk,
        input  wire        rst_n,
        input  wire        valid_in,
        input  wire signed [7:0] data_in,
        output wire [(6*8)-1:0] conv_out_flat,
        output wire        valid_out_conv
    );

    wire [(9*8)-1:0] win_flat;
    wire valid_out_buf;

    conv1_buf u_conv1_buf
        (
            .clk(clk), .rst_n(rst_n),
            .valid_in(valid_in), .data_in(data_in),
            .win_flat(win_flat), .valid_out_buf(valid_out_buf)
        );

    conv1_calc #(.DATA_BITS(8)) u_conv1_calc
        (
            .clk(clk), .rst_n(rst_n),
            .valid_out_buf(valid_out_buf), .win_flat(win_flat),
            .conv_out_flat(conv_out_flat), .valid_out_calc(valid_out_conv)
        );

endmodule
