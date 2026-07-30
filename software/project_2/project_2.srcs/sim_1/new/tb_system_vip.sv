`timescale 1ns / 1ps

import zynq_ultra_ps_e_vip_v1_0_22_pkg::*; 

module tb_system_vip();

    reg tb_ACLK;
    reg tb_ARESETn;

    initial begin
        tb_ACLK = 1'b0;
        forever #5 tb_ACLK = ~tb_ACLK;
    end

    // Goi toan bo he thong Block Design
    Memory_architectute_wrapper DUT ();

    // Khai bao Agent dung chuan phien ban v1_0_22
    zynq_ultra_ps_e_vip_v1_0_22_mst_t  ps_agent;

    // Kich ban test code C bare-metal
    initial begin
        $timeformat(-9, 0, "", 0);
        
        // Khoi tao Agent tro thang vao nhan Zynq trong Block Design
        ps_agent = new("ps_agent", DUT.Memory_architectute_i.zynq_ultra_ps_e_0.inst.IF);
        ps_agent.start_master();

        // Reset he thong
        tb_ARESETn = 1'b0;
        #200;
        tb_ARESETn = 1'b1;
        #500;

        $display("\n==================================================================");
        $display("   [SOFTWARE SIMULATION] TEST BARE-METAL C CODE DIEU KHIEN DMA    ");
        $display("==================================================================\n");

        // Ghi truc tiep vao thanh ghi DMA qua dia chi Base Address 0xB0000000
        $display("[%0t ns] [C Code] Xil_Out32(MM2S_DMACR, 0x00000001); -> KHOI DONG KENH TRUYEN ANH", $time);
        ps_agent.write_burst(32'hB0000000, 32'h00000001); 
        #50;

        $display("[%0t ns] [C Code] Xil_Out32(MM2S_SA, 0x10000000); -> CAU HINH DIA CHI CHUA ANH TRONG RAM", $time);
        ps_agent.write_burst(32'hB0000018, 32'h10000000); 
        #50;

        $display("[%0t ns] [C Code] Xil_Out32(MM2S_LENGTH, 784); -> BAT DAU TRUYEN 784 BYTES XUONG CNN", $time);
        ps_agent.write_burst(32'hB0000028, 32'd784); 
        #200;

        $display("\n==================================================================");
        $display("   HE THONG PHAN CUNG DA NHAN DUOC LENH TU PHAN MEM!   ");
        $display("==================================================================\n");

        #500;
        $finish;
    end

endmodule