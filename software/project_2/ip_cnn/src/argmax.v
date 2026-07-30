`timescale 1ns / 1ps
// =============================================================
// ARGMAX : Tim chi so lop co gia tri logit lon nhat (47 lop)
// =============================================================
module argmax47 (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    output wire        ready,
    output reg         done,

    // Cong ghi vector logit dau vao (47 phan tu, 32-bit co dau)
    input  wire        in_wr_en,
    input  wire [5:0]  in_wr_addr,
    input  wire signed [31:0] in_wr_data,

    output reg  [5:0]  predicted_class,
    output reg         valid_out
);

    reg signed [31:0] fc3_out [0:46];

    always @(posedge clk) begin
        if (in_wr_en) fc3_out[in_wr_addr] <= in_wr_data;
    end

    localparam IDLE = 0, INIT = 1, COMP = 2;
    reg [1:0] state;

    reg [15:0] c;
    reg signed [31:0] max_val;
    reg [5:0] max_idx;

    assign ready = (state == IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= IDLE;
            done            <= 0;
            valid_out       <= 0;
            predicted_class <= 0;
        end else begin
            done      <= 0;
            valid_out <= 0;
            case (state)
                IDLE: begin
                    if (start) begin
                        c <= 1;
                        state <= INIT;
                    end
                end

                INIT: begin
                    max_val <= fc3_out[0];
                    max_idx <= 0;
                    state   <= COMP;
                end

                COMP: begin
                    if (fc3_out[c] > max_val) begin
                        max_val <= fc3_out[c];
                        max_idx <= c[5:0];
                    end

                    if (c == 46) begin
                        predicted_class <= (fc3_out[c] > max_val) ? c[5:0] : max_idx;
                        valid_out <= 1;
                        done      <= 1;
                        state     <= IDLE;
                    end else begin
                        c <= c + 1;
                    end
                end
            endcase
        end
    end
endmodule
