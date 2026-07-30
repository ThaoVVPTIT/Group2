`timescale 1ns / 1ps
/*------------------------------------------------------------------------
 *  Module: fc_stream
 *  Design : "Vo streaming" bao quanh 3 tang Fully-Connected (400->120->
 *           84->47) va Argmax. Ben trong TAI SU DUNG nguyen ven 4 module
 *           da duoc kiem chung rieng le o project lenet5_modular truoc
 *           do (fc1_seq, fc2_seq, fc3_seq, argmax47 - giu dung co che
 *           mult/shift int8), chi them:
 *             1) Mot bo dem thu thap (COLLECT) nhan 16 kenh streaming tu
 *                maxpool_relu (tang pool2, 5x5x16), lam phang (FLATTEN)
 *                thanh vector 400 phan tu theo dung thu tu c*25+y*5+x.
 *             2) Mot FSM dieu phoi (giong lenet5_top_structural.v) chay
 *                lan luot fc1_seq -> fc2_seq -> fc3_seq -> argmax47,
 *                copy du lieu qua cong ghi/doc bo nho giua cac tang.
 *------------------------------------------------------------------------*/
module fc_stream
    (
        input  wire        clk,
        input  wire        rst_n,
        input  wire        valid_in,               // tu maxpool_relu (pool2), 16 kenh/xung
        input  wire [(16*8)-1:0] data_in_flat,
        output wire        ready,                  // =1 khi dang cho frame moi (IDLE)
        output reg  [5:0]  predicted_class,
        output reg         valid_out
    );

    wire signed [7:0] data_in [0:15];
    genvar gi_din;
    generate
        for (gi_din = 0; gi_din < 16; gi_din = gi_din + 1) begin : gen_din_unpack
            assign data_in[gi_din] = data_in_flat[gi_din*8 +: 8];
        end
    endgenerate

    // ------------------------------------------------------------------
    // 1) Bo dem thu thap + FLATTEN (5x5x16 -> 400)
    // ---------------------------------------------------------
    reg signed [7:0] flat_buf [0:399];
    integer pos, ch;

    reg collecting;
    reg [4:0] pos_cnt;   // 0..24 (5x5 = 25 vi tri)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pos_cnt <= 0;
        end else if (collecting && valid_in) begin
            for (ch = 0; ch < 16; ch = ch + 1)
                flat_buf[ch*25 + pos_cnt] <= data_in[ch];
            pos_cnt <= pos_cnt + 1'b1;
        end else if (!collecting) begin
            pos_cnt <= 0;
        end
    end

    // ---------------------------------------------------------
    // 2) Cac tang FC + Argmax (tai su dung nguyen ven)
    // ---------------------------------------------------------
    reg  f1_start, f2_start, f3_start, am_start;
    wire f1_done,  f2_done,  f3_done,  am_done;

    reg         f1_in_wr_en; reg [8:0] f1_in_wr_addr; reg signed [7:0] f1_in_wr_data;
    reg         f2_in_wr_en; reg [6:0] f2_in_wr_addr; reg signed [7:0] f2_in_wr_data;
    reg         f3_in_wr_en; reg [6:0] f3_in_wr_addr; reg signed [7:0] f3_in_wr_data;
    reg         am_in_wr_en; reg [5:0] am_in_wr_addr; reg signed [31:0] am_in_wr_data;

    reg  [6:0] f1_out_rd_addr; wire signed [7:0]  f1_out_rd_data;
    reg  [6:0] f2_out_rd_addr; wire signed [7:0]  f2_out_rd_data;
    reg  [5:0] f3_out_rd_addr; wire signed [31:0] f3_out_rd_data;

    fc1_seq u_fc1 (.clk(clk), .rst_n(rst_n), .start(f1_start), .ready(), .done(f1_done),
        .in_wr_en(f1_in_wr_en), .in_wr_addr(f1_in_wr_addr), .in_wr_data(f1_in_wr_data),
        .out_rd_addr(f1_out_rd_addr), .out_rd_data(f1_out_rd_data));

    fc2_seq u_fc2 (.clk(clk), .rst_n(rst_n), .start(f2_start), .ready(), .done(f2_done),
        .in_wr_en(f2_in_wr_en), .in_wr_addr(f2_in_wr_addr), .in_wr_data(f2_in_wr_data),
        .out_rd_addr(f2_out_rd_addr), .out_rd_data(f2_out_rd_data));

    fc3_seq u_fc3 (.clk(clk), .rst_n(rst_n), .start(f3_start), .ready(), .done(f3_done),
        .in_wr_en(f3_in_wr_en), .in_wr_addr(f3_in_wr_addr), .in_wr_data(f3_in_wr_data),
        .out_rd_addr(f3_out_rd_addr), .out_rd_data(f3_out_rd_data));

    wire [5:0] am_predicted_class;
    wire       am_valid_out;
    argmax47 u_argmax (.clk(clk), .rst_n(rst_n), .start(am_start), .ready(), .done(am_done),
        .in_wr_en(am_in_wr_en), .in_wr_addr(am_in_wr_addr), .in_wr_data(am_in_wr_data),
        .predicted_class(am_predicted_class), .valid_out(am_valid_out));

    // ---------------------------------------------------------
    // 3) FSM dieu phoi toan bo qua trinh (giong lenet5_top_structural.v)
    // ---------------------------------------------------------
    localparam
        IDLE      = 0,
        COLLECT   = 1,
        WR_F1_A   = 2, WR_F1_W = 3,
        RUN_F1    = 4,
        CP_F2_A   = 5, CP_F2_W = 6,
        RUN_F2    = 7,
        CP_F3_A   = 8, CP_F3_W = 9,
        RUN_F3    = 10,
        CP_AM_A   = 11, CP_AM_W = 12,
        RUN_AM    = 13,
        FINISH    = 14;

    reg [3:0] state;
    reg [15:0] cnt;

    assign ready = (state == IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            collecting <= 1;    // san sang thu thap ngay khi het reset (khong can
                                 // cho 1 chu ky IDLE de gia tri nay lan truyen,
                                 // neu khong xung valid_in DAU TIEN se bi mat)
            valid_out <= 0;
            predicted_class <= 0;
            {f1_start,f2_start,f3_start,am_start} <= 0;
            {f1_in_wr_en,f2_in_wr_en,f3_in_wr_en,am_in_wr_en} <= 0;
        end else begin
            valid_out <= 0;
            f1_start <= 0; f2_start <= 0; f3_start <= 0; am_start <= 0;
            f1_in_wr_en <= 0; f2_in_wr_en <= 0; f3_in_wr_en <= 0; am_in_wr_en <= 0;

            case (state)
                IDLE: begin
                    collecting <= 1;
                    if (valid_in) state <= COLLECT;
                end

                COLLECT: begin
                    if (pos_cnt == 25) begin  // da ghi xong du 25 vi tri (0..24)
                        collecting <= 0;
                        cnt <= 0;
                        state <= WR_F1_A;
                    end
                end

                // Copy flat_buf (400) -> fc1 in
                WR_F1_A: begin state <= WR_F1_W; end
                WR_F1_W: begin
                    f1_in_wr_en   <= 1;
                    f1_in_wr_addr <= cnt[8:0];
                    f1_in_wr_data <= flat_buf[cnt];
                    if (cnt == 399) begin f1_start <= 1; state <= RUN_F1; end
                    else begin cnt <= cnt + 1; state <= WR_F1_A; end
                end

                RUN_F1: if (f1_done) begin cnt <= 0; f1_out_rd_addr <= 0; state <= CP_F2_A; end

                CP_F2_A: begin f1_out_rd_addr <= cnt[6:0]; state <= CP_F2_W; end
                CP_F2_W: begin
                    f2_in_wr_en   <= 1;
                    f2_in_wr_addr <= cnt[6:0];
                    f2_in_wr_data <= f1_out_rd_data;
                    if (cnt == 119) begin f2_start <= 1; state <= RUN_F2; end
                    else begin cnt <= cnt + 1; state <= CP_F2_A; end
                end

                RUN_F2: if (f2_done) begin cnt <= 0; state <= CP_F3_A; end

                CP_F3_A: begin f2_out_rd_addr <= cnt[6:0]; state <= CP_F3_W; end
                CP_F3_W: begin
                    f3_in_wr_en   <= 1;
                    f3_in_wr_addr <= cnt[6:0];
                    f3_in_wr_data <= f2_out_rd_data;
                    if (cnt == 83) begin f3_start <= 1; state <= RUN_F3; end
                    else begin cnt <= cnt + 1; state <= CP_F3_A; end
                end

                RUN_F3: if (f3_done) begin cnt <= 0; state <= CP_AM_A; end

                CP_AM_A: begin f3_out_rd_addr <= cnt[5:0]; state <= CP_AM_W; end
                CP_AM_W: begin
                    am_in_wr_en   <= 1;
                    am_in_wr_addr <= cnt[5:0];
                    am_in_wr_data <= f3_out_rd_data;
                    if (cnt == 46) begin am_start <= 1; state <= RUN_AM; end
                    else begin cnt <= cnt + 1; state <= CP_AM_A; end
                end

                RUN_AM: if (am_done) state <= FINISH;

                FINISH: begin
                    predicted_class <= am_predicted_class;
                    valid_out <= 1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
