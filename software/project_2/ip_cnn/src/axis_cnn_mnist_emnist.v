`timescale 1ns / 1ps
/*------------------------------------------------------------------------
 *  Module: axis_cnn_mnist_emnist
 *  Design : Top AXI-Stream cho CNN EMNIST 3x3 (LeNet-5 rut gon), theo
 *           dung tinh than kien truc streaming cua axis_cnn_mnist.v goc
 *           trong project 20260614_RunOnChip, nhung:
 *             - Thay vi dung mot bo dem "cnt_sequencer_reg" voi cac
 *               "so ma thuat" (magic number) tinh tay cho toan bo pipeline
 *               (kho bao tri, de sai khi doi kich thuoc lop), o day dung
 *               HANDSHAKE valid/ready THAT tai tung tang - moi tang tu
 *               dem so luong dau vao/dau ra cua chinh no (conv_line_buf tu
 *               "flush" noi bo, maxpool_relu tu dong bo hang/cot). Top-level
 *               chi can dem SO PIXEL THAT (784) de biet luc nao ngung nhan
 *               frame moi, khong can biet chi tiet do tre pipeline ben trong.
 *             - 6/16 kenh (thay vi 3 kenh co dinh), 47 lop dau ra (thay vi
 *               10), kernel 3x3 (thay vi 5x5), giu nguyen co che luong tu
 *               hoa mult/shift int8.
 *
 *  Luong du lieu (khop so do khoi CNN Accelerator ban ve):
 *   s_axis (784 pixel, 1 kenh)
 *     -> conv1_layer (1->6 kenh, 3x3)      = "PE Systolic Array" lan 1
 *     -> maxpool_relu (6 kenh)             = "16x Max Pooling & ReLU" (rut gon 6)
 *     -> conv2_layer (6->16 kenh, 3x3)     = "PE Systolic Array" lan 2 (tai su dung y tuong)
 *     -> maxpool_relu (16 kenh)            = "16x Max Pooling & ReLU"
 *     -> fc_stream (FLATTEN + FC 400-120-84-47 + Argmax)
 *     -> m_axis (1 beat: predicted_class)
 *------------------------------------------------------------------------*/
module axis_cnn_mnist_emnist
    (
        input  wire        aclk,
        input  wire        aresetn,

        // ---- AXI-Stream SLAVE: nhan anh vao (784 pixel, int8) ----
        output wire        s_axis_tready,
        input  wire [7:0]  s_axis_tdata,
        input  wire        s_axis_tvalid,

        // ---- AXI-Stream MASTER: xuat ket qua du doan (1 beat) ----
        input  wire        m_axis_tready,
        output wire [7:0]  m_axis_tdata,
        output wire        m_axis_tvalid,
        output wire        m_axis_tlast
    );

    localparam IMG_PIXELS = 784;

    wire rst_n = aresetn;
    wire clk   = aclk;

    // ---------------------------------------------------------
    // 1) Dieu khien nhan frame dau vao (chi dem so pixel THAT,
    //    khong can biet do tre pipeline ben trong)
    // ---------------------------------------------------------
    reg [10:0] px_cnt;
    reg        busy;   // 1: dang xu ly 1 frame (tu luc nhan pixel dau tien
                        //    den luc fc_stream tra ve valid_out)

    wire s_axis_fire = s_axis_tvalid && s_axis_tready;
    // BUG FIX: ban goc dung "assign s_axis_tready = !busy;" khien chi
    // nhan duoc 1 pixel roi dung (busy=1 ngay sau pixel dau tien, tat
    // tready). Sua lai: cho phep nhan pixel khi CHUA du 784 pixel that
    // cua frame hien tai, HOAC khi chua bat dau frame nao (!busy).
    // Sau khi da nhan du 784 pixel, tready=0 de khong nhan them; chi
    // cho phep frame moi khi fc_stream tra ve ket qua (busy=0).
    assign s_axis_tready = !busy || (px_cnt < IMG_PIXELS);

    // s_axis_tdata is already in range [0, 127] (pixel >> 1)
    wire signed [7:0] pixel_in = s_axis_tdata;

    // ---------------------------------------------------------
    // 2) Conv1 (1 -> 6 kenh, 3x3)
    // ---------------------------------------------------------
    wire [(6*8)-1:0] c1_out_flat;
    wire c1_valid;

    conv1_layer u_conv1 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(s_axis_fire), .data_in(pixel_in),
        .conv_out_flat (c1_out_flat), .valid_out_conv(c1_valid)
    );

    // ---------------------------------------------------------
    // 3) Pool1 + ReLU (6 kenh, 26x26 -> 13x13)
    // ---------------------------------------------------------
    wire [(6*8)-1:0] p1_out_flat;
    wire p1_valid;

    maxpool_relu #(.CH(6), .DATA_BITS(8), .WIDTH(26), .HEIGHT(26)) u_pool1 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(c1_valid),        .conv_out_flat(c1_out_flat),
        .max_value_flat(p1_out_flat), .valid_out_relu(p1_valid)
    );

    // ---------------------------------------------------------
    // 4) Conv2 (6 -> 16 kenh, 3x3)
    // ---------------------------------------------------------
    wire [(16*8)-1:0] c2_out_flat;
    wire c2_valid;

    conv2_layer u_conv2 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(p1_valid),        .data_in_flat    (p1_out_flat),
        .conv2_out_flat  (c2_out_flat), .valid_out_conv2(c2_valid)
    );

    // ---------------------------------------------------------
    // 5) Pool2 + ReLU (16 kenh, 11x11(le) -> 5x5)
    // ---------------------------------------------------------
    wire [(16*8)-1:0] p2_out_flat;
    wire p2_valid;

    maxpool_relu #(.CH(16), .DATA_BITS(8), .WIDTH(11), .HEIGHT(11)) u_pool2 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(c2_valid),        .conv_out_flat(c2_out_flat),
        .max_value_flat(p2_out_flat), .valid_out_relu(p2_valid)
    );

    // ---------------------------------------------------------
    // 6) FLATTEN + FC(400->120->84->47) + Argmax
    // ---------------------------------------------------------
    wire fc_ready;
    wire [5:0] predicted_class;
    wire fc_valid_out;

    fc_stream u_fc (
        .clk(clk), .rst_n(rst_n),
        .valid_in(p2_valid), .data_in_flat(p2_out_flat),
        .ready(fc_ready),
        .predicted_class(predicted_class), .valid_out(fc_valid_out)
    );

    // ---------------------------------------------------------
    // 7) FSM "busy" muc top-level: chi theo doi frame hien tai
    // ---------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            px_cnt <= 0;
            busy   <= 0;
        end else begin
            if (!busy) begin
                if (s_axis_fire) begin
                    px_cnt <= 1;
                    busy   <= 1;
                end
            end else begin
                if (s_axis_fire && px_cnt < IMG_PIXELS) px_cnt <= px_cnt + 1'b1;
                if (fc_valid_out) busy <= 0;   // ca pipeline da tra ve ket qua
            end
        end
    end

    // s_axis_tready cho phep nhan pixel khi:
    //   - Chua bat dau frame (!busy), HOAC
    //   - Dang nhan frame nhung chua du 784 pixel (px_cnt < 784)
    // Sau khi du 784 pixel: tready=0, pipeline tu xu ly (conv_line_buf
    // tu dong flush, pool/fc tuan tu chay), cho den khi fc_valid_out
    // bao hieu ket qua xong -> busy=0 -> san sang frame moi.

    // ---------------------------------------------------------
    // 8) AXI-Stream MASTER: xuat ket qua (1 beat, tlast=tvalid)
    // ---------------------------------------------------------
    assign m_axis_tdata  = {2'b00, predicted_class};
    assign m_axis_tvalid = fc_valid_out;
    assign m_axis_tlast  = fc_valid_out;

endmodule
