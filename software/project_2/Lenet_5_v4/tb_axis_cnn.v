`timescale 1ns / 1ps // Force re-elaborate
//--------------------------------------------------------------------
// Testbench: tb_axis_cnn
// Verify toan bo pipeline CNN EMNIST 3x3 (axis_cnn_mnist_emnist)
//
// CACH CHAY:
//   (1) Icarus Verilog (can ho tro SystemVerilog):
//       cd "d:\THUC TAP\New folder"
//       iverilog -g2012 -o sim.vvp tb_axis_cnn.v axis_cnn_mnist_emnist.v \
//         conv1_layer.v conv1_buf.v conv1_calc.v conv2_layer.v conv2_buf.v \
//         conv2_calc.v conv_line_buf.v conv_calc.v maxpool_relu.v \
//         fc_stream.v fc1_seq.v fc2_seq.v fc3_seq.v argmax.v
//       vvp sim.vvp
//
//   (2) Vivado xsim:
//       cd "d:\THUC TAP\New folder"
//       xvlog --sv *.v
//       xelab tb_axis_cnn -s sim
//       xsim sim -R
//
//   (3) ModelSim / QuestaSim:
//       vlog -sv *.v
//       vsim -c tb_axis_cnn -do "run -all"
//
// LUU Y: File $readmemh dung duong dan tuong doi (weights_hex/...),
//         can chay tu thu muc chua weights_hex/ va test_image.mem
//--------------------------------------------------------------------
module tb_axis_cnn;

    //================================================================
    // Clock & Reset
    //================================================================
    reg clk = 0;
    always #5 clk = ~clk;  // 100 MHz, T = 10 ns

    reg aresetn;

    //================================================================
    // AXI-Stream Slave interface (DUT input)
    //================================================================
    reg  [7:0] s_tdata;
    reg        s_tvalid;
    wire       s_tready;

    //================================================================
    // AXI-Stream Master interface (DUT output)
    //================================================================
    wire [7:0] m_tdata;
    wire       m_tvalid;
    wire       m_tlast;
    reg        m_tready;

    //================================================================
    // DUT: axis_cnn_mnist_emnist (top-level)
    //================================================================
    axis_cnn_mnist_emnist DUT (
        .aclk          (clk),
        .aresetn       (aresetn),
        .s_axis_tready (s_tready),
        .s_axis_tdata  (s_tdata),
        .s_axis_tvalid (s_tvalid),
        .m_axis_tready (m_tready),
        .m_axis_tdata  (m_tdata),
        .m_axis_tvalid (m_tvalid),
        .m_axis_tlast  (m_tlast)
    );

    //================================================================
    // Test image storage (784 pixel = 28x28, int8)
    //================================================================
    reg [7:0] image [0:783];

    initial begin
        $readmemh("test_image.mem", image);
        #1;
        $display("DEBUG: image[0] = %x", image[0]);
        $display("DEBUG: conv1 bias[0] = %x", DUT.u_conv1.u_conv1_calc.u_calc.bias[0]);
        $display("DEBUG: conv1 kernel[0][0] = %x", DUT.u_conv1.u_conv1_calc.u_calc.kernel[0][0]);
    end

    //================================================================
    // Pipeline monitors — dem so luong valid pulses tai moi tang
    //================================================================
    integer px_count  = 0;   // so pixel da gui thanh cong (fire)
    integer c1_count  = 0;   // Conv1 valid outputs
    integer p1_count  = 0;   // Pool1 valid outputs
    integer c2_count  = 0;   // Conv2 valid outputs
    integer p2_count  = 0;   // Pool2 valid outputs
    integer clk_count = 0;   // tong clock cycles (sau reset)

    always @(posedge clk) begin
        if (aresetn) begin
            clk_count = clk_count + 1;
            if (s_tvalid && s_tready) px_count = px_count + 1;
            if (DUT.c1_valid)         c1_count = c1_count + 1;
            if (DUT.p1_valid)         p1_count = p1_count + 1;
            if (DUT.c2_valid)         c2_count = c2_count + 1;
            if (DUT.p2_valid)         p2_count = p2_count + 1;
        end
    end

    //================================================================
    // FC state monitor — hien thi khi co chuyen doi trang thai
    //================================================================
    reg [3:0] fc_state_prev = 4'hF;

    always @(posedge clk) begin
        if (aresetn && DUT.u_fc.state !== fc_state_prev) begin
            case (DUT.u_fc.state)
                4'd0:  $display("[%0t] FC: IDLE",                 $time);
                4'd1:  $display("[%0t] FC: COLLECT (5x5x16->400)", $time);
                4'd4:  $display("[%0t] FC: RUN_F1 (400->120)...",  $time);
                4'd7:  $display("[%0t] FC: RUN_F2 (120->84)...",   $time);
                4'd10: $display("[%0t] FC: RUN_F3 (84->47)...",    $time);
                4'd13: $display("[%0t] FC: RUN_AM (argmax47)...",  $time);
                4'd14: $display("[%0t] FC: FINISH",                $time);
            endcase
            fc_state_prev <= DUT.u_fc.state;
        end
    end

    //================================================================
    // Key event milestones — bao khi tung tang bat dau/ket thuc
    //================================================================
    reg c1_started = 0, p1_started = 0, c2_started = 0, p2_started = 0;

    always @(posedge clk) begin
        if (aresetn) begin
            if (DUT.c1_valid && !c1_started) begin
                $display("[%0t] Conv1: first valid output (after %0d clocks)", $time, clk_count);
                c1_started <= 1;
            end
            if (DUT.p1_valid && !p1_started) begin
                $display("[%0t] Pool1: first valid output", $time);
                p1_started <= 1;
            end
            if (DUT.c2_valid && !c2_started) begin
                $display("[%0t] Conv2: first valid output", $time);
                c2_started <= 1;
            end
            if (DUT.p2_valid && !p2_started) begin
                $display("[%0t] Pool2: first valid output", $time);
                p2_started <= 1;
            end
        end
    end

    //================================================================
    // Main test sequence
    //================================================================
    integer i;
    reg [63:0] t_start, t_end;

    initial begin
        // Optional: dump waveform (uncomment for debugging)
        // $dumpfile("tb_axis_cnn.vcd");
        // $dumpvars(0, tb_axis_cnn);

        $display("");
        $display("============================================================");
        $display("  CNN EMNIST 3x3 Pipeline — Full Integration Testbench");
        $display("  Image: 28x28 int8, Model: Conv1->Pool1->Conv2->Pool2->FC");
        $display("  Output: 47 classes, predicted_class (6-bit)");
        $display("============================================================");
        $display("");

        // ---- Khoi tao ----
        aresetn  = 0;
        s_tvalid = 0;
        s_tdata  = 0;
        m_tready = 1;   // luon san sang nhan ket qua

        // ---- Reset ----
        repeat (20) @(posedge clk);
        aresetn = 1;
        $display("[%0t] Reset released.", $time);
        repeat (5) @(posedge clk);

        // ---- Feed 784 pixels qua AXI-Stream ----
        $display("[%0t] Feeding 784 pixels (28x28 image)...", $time);
        t_start = $time;

        for (i = 0; i < 784; i = i + 1) begin
            s_tdata  = image[i];
            s_tvalid = 1;
            @(posedge clk);
            while (!s_tready) @(posedge clk);
        end
        @(posedge clk);
        s_tvalid = 0;

        $display("[%0t] All 784 pixels sent. px_count=%0d. Waiting for result...", $time, px_count);

        // ---- Cho ket qua tu m_axis ----
        wait (m_tvalid === 1'b1);
        @(negedge clk);   // sample on negedge to avoid race condition
        t_end = $time;

        // ---- Bao cao ket qua ----
        $display("");
        $display("============================================================");
        $display("  RESULT: predicted_class = %0d", m_tdata[5:0]);
        $display("============================================================");
        $display("");
        $display("  Pipeline Statistics:");
        $display("    Pixels fed       : %0d  (expected:  784 = 28x28)", px_count);
        $display("    Conv1 outputs    : %0d  (expected:  676 = 26x26)", c1_count);
        $display("    Pool1 outputs    : %0d  (expected:  169 = 13x13)", p1_count);
        $display("    Conv2 outputs    : %0d  (expected:  121 = 11x11)", c2_count);
        $display("    Pool2 outputs    : %0d  (expected:   25 =  5x5)",  p2_count);
        $display("    Total clocks     : %0d",                           clk_count);
        $display("    Latency (ns)     : %0d  (from first pixel to result)", t_end - t_start);
        $display("");

        // ---- Kiem tra tu dong ----
        begin : validation
            integer pass;
            pass = 1;
            if (px_count !== 784) begin
                $display("  [FAIL] Pixel count = %0d, expected 784", px_count);
                pass = 0;
            end
            if (c1_count !== 676) begin
                $display("  [FAIL] Conv1 count = %0d, expected 676 (26x26)", c1_count);
                pass = 0;
            end
            if (p1_count !== 169) begin
                $display("  [FAIL] Pool1 count = %0d, expected 169 (13x13)", p1_count);
                pass = 0;
            end
            if (c2_count !== 121) begin
                $display("  [FAIL] Conv2 count = %0d, expected 121 (11x11)", c2_count);
                pass = 0;
            end
            if (p2_count !== 25) begin
                $display("  [FAIL] Pool2 count = %0d, expected 25 (5x5)", p2_count);
                pass = 0;
            end

            if (m_tdata[5:0] > 46) begin
                $display("  [FAIL] predicted_class = %0d, out of range [0,46]!", m_tdata[5:0]);
                pass = 0;
            end

            if (m_tlast !== 1'b1) begin
                $display("  [FAIL] m_axis_tlast should be 1 when tvalid=1");
                pass = 0;
            end

            $display("");
            if (pass)
                $display("  *** ALL PIPELINE CHECKS PASSED ***");
            else
                $display("  *** SOME CHECKS FAILED — SEE ABOVE ***");
        end

        $display("---------------------------------------------------------");
        $display("DEBUG: Conv1[0] = %d", DUT.u_conv1.u_conv1_calc.conv_out_flat[7:0]);
        $display("DEBUG: Pool1[0] = %d", DUT.u_pool1.max_value_flat[7:0]);
        $display("DEBUG: Conv2[0] = %d", DUT.u_conv2.u_conv2_calc.conv_out_flat[7:0]);
        $display("DEBUG: Pool2[0] = %d", DUT.u_pool2.max_value_flat[7:0]);
        $display("DEBUG: FC1_out[0] = %d", DUT.u_fc.u_fc1.fc1_out[0]);
        $display("DEBUG: FC2_out[0] = %d", DUT.u_fc.u_fc2.fc2_out[0]);
        $display("DEBUG: FC3_out[0] = %d", DUT.u_fc.u_fc3.fc3_out[0]);
        $display("DEBUG: Argmax max_val = %d, max_idx = %d", DUT.u_fc.u_argmax.max_val, DUT.u_fc.u_argmax.max_idx);
        $display("---------------------------------------------------------");
        $display("");
        #200;
        $finish;
    end

    //================================================================
    // Timeout watchdog — prevent infinite simulation
    //================================================================
    initial begin
        #50_000_000;   // 50 ms = 5,000,000 cycles at 100MHz
        $display("");
        $display("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        $display("  TIMEOUT: Simulation exceeded 5,000,000 clock cycles!");
        $display("  Pipeline is likely stuck. Dumping partial progress:");
        $display("    Pixels fed    : %0d / 784", px_count);
        $display("    Conv1 outputs : %0d / 676", c1_count);
        $display("    Pool1 outputs : %0d / 169", p1_count);
        $display("    Conv2 outputs : %0d / 121", c2_count);
        $display("    Pool2 outputs : %0d / 25",  p2_count);
        $display("    FC state      : %0d",       DUT.u_fc.state);
        $display("    Top busy      : %0b",       DUT.busy);
        $display("    px_cnt        : %0d",       DUT.px_cnt);
        $display("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        $finish;
    end

    //================================================================
    // Progress reporting — in moi 100,000 clocks
    //================================================================
    always @(posedge clk) begin
        if (aresetn && clk_count > 0 && (clk_count % 100000 == 0)) begin
            $display("[%0t] Progress: clk=%0d px=%0d c1=%0d p1=%0d c2=%0d p2=%0d fc_state=%0d",
                $time, clk_count, px_count, c1_count, p1_count, c2_count, p2_count,
                DUT.u_fc.state);
        end
    end

endmodule
