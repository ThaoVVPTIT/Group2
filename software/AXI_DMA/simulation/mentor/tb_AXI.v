`timescale 1ns / 1ps
import avalon_mm_pkg::*;

module tb_AXI();
    logic clk = 0;
    logic reset_n = 0; 
    wire  irq;

    AXI_DMA dut (
        .clk_clk              (clk),
        .msgdma_0_csr_irq_irq (irq),
        .reset_reset_n        (reset_n)
    );

    // Xung Clock chu kỳ 20ns
    always #10 clk = ~clk; 

    task avalon_write(input logic [31:0] addr, input logic [31:0] data);
    begin
        @(posedge clk); 
        $display("-> [%0t ns] BFM GHI data %h vao dia chi %h", $time, data, addr);
        
        dut.mm_master_bfm_0.set_command_request(REQ_WRITE);
        dut.mm_master_bfm_0.set_command_idle(0, 0);
        dut.mm_master_bfm_0.set_command_init_latency(0);
        dut.mm_master_bfm_0.set_command_address(addr);
        dut.mm_master_bfm_0.set_command_byte_enable(4'hF, 0);
        dut.mm_master_bfm_0.set_command_data(data, 0);
        dut.mm_master_bfm_0.set_command_burst_count(1);
        dut.mm_master_bfm_0.set_command_burst_size(4);
        
        dut.mm_master_bfm_0.push_command();
        repeat(2) @(posedge clk); 
    end
    endtask

    initial begin
        // 1. Dìm mạch vào Reset
        reset_n = 0;      
        #100;         
        reset_n = 1; // Kéo lên 1 để nhả Reset
        
        // 2. CHỜ RESET ĐỒNG BỘ (BƯỚC QUAN TRỌNG NHẤT)
        // Cho hệ thống nghỉ hẳn 10 chu kỳ clock (200ns) để rst_controller nhả hoàn toàn
        #200; 

        $display("=== MẠCH ĐÃ TỈNH NGỦ, BẮT ĐẦU KHỞI TẠO BFM ===");

        // 3. Khởi tạo BFM SAU KHI mạch đã thực sự tỉnh
        dut.mm_master_bfm_0.init();
        dut.st_sink_bfm_0.init();
        dut.st_sink_bfm_0.set_ready(1); 
        #50; // Cho BFM 50ns để nạp cấu hình

        $display("=== HE THONG DA SAN SANG o moc %0t ns ===", $time);

        // 4. Lệnh ghi đầu tiên sẽ xuất phát ở mốc 350ns
        avalon_write(32'h0000_0000, 32'hAABBCCDD); 
        avalon_write(32'h0000_0004, 32'h11223344); 
        
        avalon_write(32'h0000_1020, 32'h0000_0000); 
        avalon_write(32'h0000_1024, 32'h0000_0000); 
        avalon_write(32'h0000_1028, 32'd8);        
        avalon_write(32'h0000_102C, 32'h8000_0000); 

        #2000; 
        $display("=== HOAN TAT MO PHONG ===");
        $stop;
    end
endmodule