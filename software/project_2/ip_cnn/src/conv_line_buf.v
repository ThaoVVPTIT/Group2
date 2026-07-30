`timescale 1ns / 1ps
/*------------------------------------------------------------------------
 *  Module: conv_line_buf
 *  Vai tro: Bo dem dong (line buffer) tao "cua so truot" FILTER x FILTER
 *           tu luong pixel den tuan tu (1 gia tri/chu ky, dung AXI-Stream
 *           style valid_in). Day la loi tai su dung chung cho ca
 *           conv1_buf.v va conv2_buf.v, viet tong quat hoa theo dung
 *           thuat toan cua conv1_buf.v goc trong project 20260614_RunOnChip
 *           (chi khac: FILTER duoc tham so hoa thay vi co dinh 5x5,
 *           va cac tap dau ra dung mang unpacked "win[]" thay vi liet ke
 *           tung tin hieu data_out_0..24 rieng le, giup tai su dung duoc
 *           cho ca FILTER=3 (kernel cua LeNet EMNIST 3x3) o nhieu WIDTH
 *           khac nhau (28 cho conv1, 13 cho conv2)).
 *------------------------------------------------------------------------*/
module conv_line_buf
    #(
        parameter WIDTH     = 28,
        parameter HEIGHT    = 28,
        parameter DATA_BITS = 8,
        parameter FILTER    = 3
    )
    (
        input  wire                        clk,
        input  wire                        rst_n,
        input  wire                        valid_in,
        input  wire signed [DATA_BITS-1:0] data_in,
        output wire [(FILTER*FILTER*DATA_BITS)-1:0] win_flat,
        output reg                         valid_out_buf
    );

    reg signed [DATA_BITS-1:0] win [0:FILTER*FILTER-1];
    genvar gi_win;
    generate
        for (gi_win = 0; gi_win < FILTER*FILTER; gi_win = gi_win + 1) begin : gen_win_flat
            assign win_flat[gi_win*DATA_BITS +: DATA_BITS] = win[gi_win];
        end
    endgenerate

    localparam OUT_W = WIDTH  - FILTER + 1;
    localparam OUT_H = HEIGHT - FILTER + 1;
    localparam TOTAL_IN = WIDTH*HEIGHT;

    reg signed [DATA_BITS-1:0] buffer [0:WIDTH*FILTER-1];
    integer buf_idx;
    integer w_idx, h_idx;
    integer buf_flag;   // 0 .. FILTER-1 : chi so dong vat ly hien la "dong logic 0"
    reg     state;      // 0: dang nap FILTER dong dau tien, 1: da du du lieu, quet cua so

    // Bo dem "flush tu dong": vi cua so truot can nhin truoc FILTER dong,
    // dong dau ra cuoi cung luon can them (WIDTH pulses) sau khi da nhan
    // du WIDTH*HEIGHT pixel that. Thay vi bat top-level phai tinh tay
    // "so ma thuat" so chu ky flush can them (nhu cnt_sequencer_reg trong
    // ban goc), module nay TU DONG sinh them cac chu ky flush noi bo
    // (auto_flush) ngay khi da nhan du so pixel that, cho den khi tu
    // ket thuc (state tro ve 0). Nho vay khong can bat ky "magic number"
    // nao o tang tren.
    integer in_cnt;
    wire auto_flush = (in_cnt >= TOTAL_IN) && (state == 1'b1);
    wire proc_en    = valid_in || auto_flush;

    integer i, r, c;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < WIDTH*FILTER; i = i + 1) buffer[i] <= 0;
            for (i = 0; i < FILTER*FILTER; i = i + 1) win[i] <= 0;
            buf_idx  <= 0;
            w_idx    <= 0;
            h_idx    <= 0;
            buf_flag <= 0;
            state    <= 0;
            in_cnt   <= 0;
            valid_out_buf <= 0;
        end else begin
            valid_out_buf <= 1'b0; // Mac dinh la 0, chi pulse 1 nhip khi co cua so hop le

            if (valid_in && in_cnt < TOTAL_IN) in_cnt <= in_cnt + 1;

            if (valid_in) begin
                buffer[buf_idx] <= data_in;
                if (buf_idx == WIDTH*FILTER-1) buf_idx <= 0;
                else                           buf_idx <= buf_idx + 1;
            end

            if (proc_en) begin
                if (state && w_idx < OUT_W) valid_out_buf <= 1'b1;

                if (!state) begin
                    // dang do FILTER dong dau tien vao buffer
                    if (buf_idx == WIDTH*FILTER-1) state <= 1'b1;
                end else begin
                    w_idx <= w_idx + 1;

                    if (w_idx == WIDTH-1) begin
                        if (buf_flag == FILTER-1) buf_flag <= 0;
                        else                       buf_flag <= buf_flag + 1;
                        w_idx <= 0;

                        if (h_idx == HEIGHT-FILTER) begin  // da quet het OUT_H dong
                            h_idx <= 0;
                            state <= 1'b0;
                        end else h_idx <= h_idx + 1;
                    end

                    // Tao cua so FILTER x FILTER: dong logic r ung voi dong vat ly
                    // (buf_flag + r) mod FILTER trong bo dem tron (circular buffer)
                    for (r = 0; r < FILTER; r = r + 1) begin
                        for (c = 0; c < FILTER; c = c + 1) begin
                            win[r*FILTER + c] <= buffer[((buf_flag + r) % FILTER)*WIDTH + w_idx + c];
                        end
                    end
                end
            end
        end
    end
endmodule
