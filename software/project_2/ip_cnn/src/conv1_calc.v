`timescale 1ns / 1ps
/*------------------------------------------------------------------------
 *  Module: conv1_calc
 *  Design : Tinh tich chap lop Conv1 (1 -> 6 kenh, kernel 3x3)
 *------------------------------------------------------------------------*/
module conv1_calc
    #(
        parameter DATA_BITS = 8
    )
    (
        input  wire clk,
        input  wire rst_n,
        input  wire valid_out_buf,
        input  wire [(9*DATA_BITS)-1:0] win_flat,
        output wire [(6*DATA_BITS)-1:0] conv_out_flat,
        output wire valid_out_calc
    );

    conv_calc
        #(
            .IN_CH(1), .OUT_CH(6), .DATA_BITS(DATA_BITS),
            .WEIGHT_FILE("conv1_kernel.mem"),
            .BIAS_FILE("conv1_bias.mem"),
            .MULT_FILE("conv1_multiplier.mem"),
            .SHIFT_FILE("conv1_shift.mem")
        )
        u_calc
        (
            .clk(clk), .rst_n(rst_n), .valid_in(valid_out_buf),
            .win_flat(win_flat),
            .conv_out_flat(conv_out_flat), .valid_out(valid_out_calc)
        );

endmodule
