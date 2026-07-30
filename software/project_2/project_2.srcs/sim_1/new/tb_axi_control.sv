`timescale 1ns / 1ps

module tb_axi_control();

    // 1. Khai báo hệ thống Block Design
    Memory_architectute_wrapper DUT ();

    // 2. Macro định tuyến tín hiệu (Giúp code gọn gàng)
    `define DMA_CLK DUT.Memory_architectute_i.axi_dma_0.s_axi_lite_aclk
    `define DMA_IP  DUT.Memory_architectute_i.axi_dma_0

    // Biến phụ trợ cho vòng lặp để tránh lỗi VRFC 10-3142
    integer pixel_idx;
    reg [31:0] temp_rdata;

    // ==================================================================
    // BFM 1: TÁC VỤ GIA LẬP CPU GHI LỆNH XUỐNG THANH GHI AXI-LITE
    // ==================================================================
    task axi_lite_write(input [31:0] addr, input [31:0] data);
    begin
        // Pha 1: Bơm địa chỉ
        force `DMA_IP.s_axi_lite_awaddr = addr;
        force `DMA_IP.s_axi_lite_awvalid = 1'b1;
        do begin @(posedge `DMA_CLK); end while (`DMA_IP.s_axi_lite_awready == 1'b0);
        force `DMA_IP.s_axi_lite_awvalid = 1'b0;

        // Pha 2: Bơm dữ liệu
        force `DMA_IP.s_axi_lite_wdata = data;
        force `DMA_IP.s_axi_lite_wvalid = 1'b1;
        do begin @(posedge `DMA_CLK); end while (`DMA_IP.s_axi_lite_wready == 1'b0);
        force `DMA_IP.s_axi_lite_wvalid = 1'b0;

        // Pha 3: Xác nhận hoàn tất
        force `DMA_IP.s_axi_lite_bready = 1'b1;
        do begin @(posedge `DMA_CLK); end while (`DMA_IP.s_axi_lite_bvalid == 1'b0);
        force `DMA_IP.s_axi_lite_bready = 1'b0;
    end
    endtask

    // ==================================================================
    // BFM 2: GIA LẬP BỘ NHỚ RAM TRẢ VỀ DỮ LIỆU ẢNH
    // ==================================================================
    initial begin
        // Khởi tạo trạng thái nghỉ
        force `DMA_IP.m_axi_mm2s_arready = 1'b1; 
        force `DMA_IP.m_axi_mm2s_rvalid  = 1'b0;
        force `DMA_IP.m_axi_mm2s_rlast   = 1'b0;
        force `DMA_IP.m_axi_mm2s_rdata   = 32'h0;
        
        forever begin
            @(posedge `DMA_CLK);
            
            // Nếu phát hiện DMA đưa địa chỉ lên đòi đọc ảnh
            if (`DMA_IP.m_axi_mm2s_arvalid == 1'b1) begin
                @(posedge `DMA_CLK);
                force `DMA_IP.m_axi_mm2s_rvalid = 1'b1; 
                
                // Bơm dữ liệu (Fix lỗi VRFC 10-3142 bằng biến tĩnh)
                for (pixel_idx = 0; pixel_idx < 196; pixel_idx = pixel_idx + 1) begin
                    temp_rdata = pixel_idx;
                    force `DMA_IP.m_axi_mm2s_rdata = temp_rdata; 
                    
                    if (pixel_idx == 195) force `DMA_IP.m_axi_mm2s_rlast = 1'b1; 
                    else force `DMA_IP.m_axi_mm2s_rlast = 1'b0;
                    
                    do begin @(posedge `DMA_CLK); end while (`DMA_IP.m_axi_mm2s_rready == 1'b0);
                end
                
                // Reset tín hiệu sau khi bơm xong
                force `DMA_IP.m_axi_mm2s_rvalid = 1'b0;
                force `DMA_IP.m_axi_mm2s_rlast  = 1'b0;
                force `DMA_IP.m_axi_mm2s_rdata  = 32'h0;
            end
        end
    end

    // ==================================================================
    // CHƯƠNG TRÌNH CHÍNH (KỊCH BẢN ĐIỀU KHIỂN)
    // ==================================================================
    initial begin
        $timeformat(-9, 0, "", 0);
        
        #1500; // Đợi ổn định ban đầu

        $display("\n==================================================================");
        $display("   [BFM TESTBENCH] BAT DAU KICH HOAT TRUYEN ANH QUA AXI DMA       ");
        $display("==================================================================\n");

        $display("[%0t ns] [AXI-Lite] 1. Set Source Address (0x10000000)", $time);
        axi_lite_write(32'hB0000018, 32'h10000000); 
        #100;

        $display("[%0t ns] [AXI-Lite] 2. Run DMA (Control Register)", $time);
        axi_lite_write(32'hB0000000, 32'h00000001); 
        #100;

        $display("[%0t ns] [AXI-Lite] 3. Set Transfer Length (784 Bytes)", $time);
        axi_lite_write(32'hB0000028, 32'd784); 
        
        $display("\n==================================================================");
        $display("   HOAN TAT GHI LENH! HAY KTRA RDATA VA TDATA TREN WAVEFORM       ");
        $display("==================================================================\n");

        #5000;
        $finish;
    end

endmodule