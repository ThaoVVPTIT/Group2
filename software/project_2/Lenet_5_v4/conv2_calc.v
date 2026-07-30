`timescale 1ns / 1ps
/*------------------------------------------------------------------------
 *  Module: conv2_calc
 *  Design : Tinh tich chap lop Conv2 (6 -> 16 kenh, kernel 3x3)
 *------------------------------------------------------------------------*/
module conv2_calc
    #(
        parameter DATA_BITS = 8
    )
    (
        input  wire clk,
        input  wire rst_n,
        input  wire valid_out_buf,
        input  wire [(54*DATA_BITS)-1:0] win_flat,   // 6 kenh x 9 tap
        output wire [(16*DATA_BITS)-1:0] conv_out_flat,
        output wire valid_out_calc
    );

    conv_calc
        #(
            .IN_CH(6), .OUT_CH(16), .DATA_BITS(DATA_BITS),
            .WEIGHT_FILE("conv2_kernel.mem"),
            .BIAS_FILE("conv2_bias.mem"),
            .MULT_FILE("conv2_multiplier.mem"),
            .SHIFT_FILE("conv2_shift.mem")
        )
        u_calc
        (
            .clk(clk), .rst_n(rst_n), .valid_in(valid_out_buf),
            .win_flat(win_flat),
            .conv_out_flat(conv_out_flat), .valid_out(valid_out_calc)
        );

endmodule
