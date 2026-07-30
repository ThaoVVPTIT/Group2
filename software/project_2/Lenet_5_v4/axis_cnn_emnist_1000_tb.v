`timescale 1ns / 1ps
//--------------------------------------------------------------------
// Testbench: axis_cnn_emnist_1000_tb
// Muc dich: Test 1000 anh tu bo du lieu EMNIST.
//           So sanh ket qua du doan voi nhan (label) thuc te.
//--------------------------------------------------------------------

module axis_cnn_emnist_1000_tb;

    // --- Signals cho DUT ---
    reg  clk;
    reg  aresetn;
    reg  [7:0] s_tdata;
    reg  s_tvalid;
    wire s_tready;
    wire [7:0] m_tdata;
    wire m_tvalid;

    // --- Khoi tao DUT ---
    axis_cnn_mnist_emnist DUT (
        .aclk(clk),
        .aresetn(aresetn),
        .s_axis_tdata(s_tdata),
        .s_axis_tvalid(s_tvalid),
        .s_axis_tready(s_tready),
        .m_axis_tdata(m_tdata),
        .m_axis_tvalid(m_tvalid),
        .m_axis_tready(1'b1),
        .m_axis_tlast()
    );

    // --- Clock Generator ---
    always #5 clk = ~clk;

    // --- Memory cho 1000 anh va 1000 nhan ---
    reg [7:0] images [0:783999]; // 1000 * 784
    reg [7:0] labels [0:999];

    initial begin
        $readmemh("input_1000.mem", images);
        $readmemh("label_1000.mem", labels);
    end

    // --- Variables de theo doi tien do ---
    integer img_idx, px_idx;
    integer r, c, raw_idx;
    integer correct_count;
    integer error_count;
    integer start_time;
    integer end_time;

    // Log file de ghi nhung anh bi sai
    integer log_file;

    initial begin
        // Khoi tao tin hieu
        clk = 0;
        aresetn = 0;
        s_tdata = 0;
        s_tvalid = 0;
        correct_count = 0;
        error_count = 0;

        // Cho he thong on dinh
        #100;

        log_file = $fopen("error_log.txt", "w");
        $fdisplay(log_file, "--- ERROR LOG FOR 1000 IMAGES ---");

        $display("============================================================");
        $display("   BAT DAU TEST 1000 ANH EMNIST");
        $display("============================================================");

        start_time = $time;

        for (img_idx = 0; img_idx < 1000; img_idx = img_idx + 1) begin
            
            // --- XOA TRANG PIPELINE TRUOC MOI ANH ---
            // Bang cach nhan nut Reset, chung ta dam bao cac bo dem (line buffer)
            // khong bi luu lai pixel rac (padding) tu buc anh phia truoc.
            @(posedge clk);
            s_tvalid = 0;
            aresetn = 0;
            #50;
            aresetn = 1;
            #50;
            // ----------------------------------------

            // 1. Gui 784 pixels
            // Input format: pixel_raw / 2 de map [0,255] -> [0,127]
            // (khop voi S_in_conv1 = 127.0 trong train.py,
            //  tuc la img_float * 127 voi img_float in [0,1])
            // KHONG transpose vi anh da duoc sua huong san.
            for (px_idx = 0; px_idx < 784; px_idx = px_idx + 1) begin
                @(posedge clk);
                s_tdata  = images[img_idx * 784 + px_idx] >> 1;
                s_tvalid = 1;
                // Cho den khi DUT san sang nhan (tready = 1)
                while (!s_tready) begin
                    @(posedge clk);
                end
            end
            
            // 2. Dung gui du lieu
            @(posedge clk);
            s_tvalid = 0;

            // 3. Cho ket qua tu mang CNN (m_tvalid)
            wait (m_tvalid === 1'b1);
            @(negedge clk); // Chot gia tri o suon xuong cho an toan
            
            // 4. So sanh voi nhan dung (Label)
            if (m_tdata === labels[img_idx]) begin
                correct_count = correct_count + 1;
            end else begin
                error_count = error_count + 1;
                $display("[Image %4d] ERROR  : Predicted = %d, Label = %d", img_idx, m_tdata, labels[img_idx]);
                $fdisplay(log_file, "[Image %4d] ERROR  : Predicted = %d, Label = %d", img_idx, m_tdata, labels[img_idx]);
            end
            
            // Hien thi tien do sau moi 100 anh de theo doi
            if ((img_idx + 1) % 100 == 0) begin
                $display("Progress: %0d / 1000 images processed...", img_idx + 1);
            end
        end

        end_time = $time;
        
        // --- IN KET QUA TONG KET ---
        $display("============================================================");
        $display("   KET QUA TEST 1000 ANH");
        $display("============================================================");
        $display(" - Total Images : 1000");
        $display(" - Correct      : %0d", correct_count);
        $display(" - Errors       : %0d", error_count);
        $display(" - Accuracy     : %0d.%0d %%", correct_count / 10, correct_count % 10);
        $display(" - Total Time   : %0d ns", end_time - start_time);
        $display("============================================================");
        $display("Vui long kiem tra file 'error_log.txt' trong thu muc chay mo phong cua Vivado de xem chi tiet tung anh bi sai.");
        
        $fclose(log_file);
        $finish;
    end

endmodule
