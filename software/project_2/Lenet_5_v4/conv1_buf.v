`timescale 1ns / 1ps
/*------------------------------------------------------------------------
 *  Module: conv1_buf
 *  Design : Bo dem dau vao cho lop tich chap thu nhat (Conv1)
 *           Anh vao: 1 x 28 x 28, kernel 3x3 -> cua so truot 9 tap
 *           (theo dung "form" cua conv1_buf.v trong project
 *           20260614_RunOnChip, tai su dung loi conv_line_buf.v)
 *------------------------------------------------------------------------*/
module conv1_buf
    #(
        parameter WIDTH     = 28,
        parameter HEIGHT    = 28,
        parameter DATA_BITS = 8
    )
    (
        input  wire                        clk,
        input  wire                        rst_n,
        input  wire                        valid_in,
        input  wire signed [DATA_BITS-1:0] data_in,
        output wire [(9*DATA_BITS)-1:0] win_flat,   // 3x3 = 9 tap
        output wire                        valid_out_buf
    );

    conv_line_buf
        #(
            .WIDTH(WIDTH), .HEIGHT(HEIGHT), .DATA_BITS(DATA_BITS), .FILTER(3)
        )
        u_line_buf
        (
            .clk(clk), .rst_n(rst_n),
            .valid_in(valid_in), .data_in(data_in),
            .win_flat(win_flat), .valid_out_buf(valid_out_buf)
        );

endmodule
