`timescale 1ns / 1ps

module tb_cnn_top();

    // -----------------------------------------------------------
    // 1. Khai bao tin hieu
    // -----------------------------------------------------------
    reg        aclk;
    reg        aresetn;

    // AXI-Stream Slave (Chieu day anh vao CNN)
    wire       s_axis_tready;
    reg  [7:0] s_axis_tdata;
    reg        s_axis_tvalid;

    // AXI-Stream Master (Chieu nhan ket qua tu CNN)
    reg        m_axis_tready;
    wire [7:0] m_axis_tdata;
    wire       m_axis_tvalid;
    wire       m_axis_tlast;

    // -----------------------------------------------------------
    // 2. Khoi tao module DUT (Device Under Test)
    // -----------------------------------------------------------
    axis_cnn_mnist_emnist dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tlast(m_axis_tlast)
    );

    // -----------------------------------------------------------
    // 3. Tao xung nhip (Clock 100MHz = Chu ky 10ns)
    // -----------------------------------------------------------
    initial begin
        aclk = 0;
        forever #5 aclk = ~aclk; 
    end

    // -----------------------------------------------------------
    // 4. Bien phu & Bien Do luong Hieu nang (Performance)
    // -----------------------------------------------------------
    integer i;
    reg [7:0] test_image [0:783]; 
    
    // Bien do luong thoi gian va xung dot de danh gia kien truc
    time start_time;
    time end_time;
    integer clock_cycles_latency;
    integer backpressure_stalls;

    // -----------------------------------------------------------
    // 5. Kich ban Test (Test Scenario)
    // -----------------------------------------------------------
    initial begin
        // THIET LAP FORMAT THOI GIAN: 
        // -9 (nanosecond), 0 (khong co so thap phan), "" (khong dung hau to tu dong), 0 (khoang trang)
        $timeformat(-9, 0, "", 0);

        // Khoi tao tin hieu mac dinh
        aresetn       = 0;
        s_axis_tvalid = 0;
        s_axis_tdata  = 0;
        m_axis_tready = 1; // DMA ao luon san sang nhan ket qua
        backpressure_stalls = 0; 

        // Tao du lieu gia lap de test hieu nang phan cung (Dummy data)
        for (i = 0; i < 784; i = i + 1) begin
            test_image[i] = (i * 3) % 127;
        end

        // Reset mach (doi 4 chu ky roi tha reset)
        #40;
        aresetn = 1;
        #20;

        $display("\n=======================================================");
        $display("   [EVALUATION] KIEM TRA HIEU NANG KIEN TRUC BO NHO    ");
        $display("   -> Focus 1: Continuous Data Supply (Stalls)         ");
        $display("   -> Focus 2: Processing & Memory Access Latency      ");
        $display("=======================================================\n");

        // Ghi nhan moc thoi gian bat dau dua pixel dau tien vao
        start_time = $time;

        // BOM 784 PIXEL VAO MACH (Gia lap DMA S2MM)
        for (i = 0; i < 784; i = i + 1) begin
            
            s_axis_tdata  = test_image[i];
            s_axis_tvalid = 1; 

            // --- KIEM TRA KHA NANG CAP DU LIEU LIEN TUC ---
            // Neu CNN chua san sang (tready=0) ma valid da len 1 -> Bi stall (Nghen)
            if (s_axis_tready == 0) begin
                backpressure_stalls = backpressure_stalls + 1;
            end

            // Doi den khi CNN keo tready len 1 (dong y nhan)
            wait(s_axis_tready == 1);
            
            // Doi dung 1 suon len cua clock de Handshake xay ra
            @(posedge aclk);
        end

        // Ha valid xuong vi da truyen xong khung anh
        s_axis_tvalid = 0;
        $display("-> [Thoi gian: %0t ns] Da cap xong 784 bytes du lieu. Dang doi xu ly...", $time);

        // -----------------------------------------------------------
        // 6. Cho doi ket qua tra ve & Tinh toan Latency
        // -----------------------------------------------------------
        wait(m_axis_tvalid == 1 && m_axis_tready == 1);
        
        // Ghi nhan moc thoi gian nhan duoc ket qua cuoi cung
        end_time = $time;
        
        // Tinh toan do tre (Chu ky clock = 10ns)
        clock_cycles_latency = (end_time - start_time) / 10;
        
        @(posedge aclk); 

        // -----------------------------------------------------------
        // 7. IN BAO CAO HIEU NANG TRONG TAM RA TCL CONSOLE
        // -----------------------------------------------------------
        $display("\n*******************************************************");
        $display("               BAO CAO DO LUONG HIEU NANG              ");
        $display("*******************************************************");
        
        $display(" [1] KHA NANG CAP DU LIEU LIEN TUC (DATA SUPPLY):");
        $display("     -> Tong so du lieu da nap : 784 bytes");
        $display("     -> So chu ky bi nghen     : %0d chu ky (stalls)", backpressure_stalls);
        if (backpressure_stalls == 0)
            $display("     -> DANH GIA: Kien truc Buffer hoat dong hoan hao. Mach co kha nang nhan\n                  du lieu lien tuc 1 byte/clock ma khong tao ra Backpressure.");
        else
            $display("     -> DANH GIA: Mach bi nghen co chai, bo dem (Buffer) chua du sau.");
            
        $display("-------------------------------------------------------");
        $display(" [2] DO TRE XU LY & TRUY XUAT BO NHO (LATENCY):");
        $display("     -> Moc thoi gian bat dau  : %0t ns", start_time);
        $display("     -> Moc thoi gian ket thuc : %0t ns", end_time);
        $display("     -> TONG DO TRE (LATENCY)  : %0d ns", (end_time - start_time));
        $display("     -> TUONG DUONG            : %0d chu ky Clock", clock_cycles_latency);
        $display("     -> DANH GIA: Thoi gian nay phan anh toc do cua toan bo Pipeline,");
        $display("                  bao gom qua trinh tich chap va truy xuat trong so/bias");
        $display("                  tu kien truc bo nho (Registers/LUTRAM).");
        $display("*******************************************************\n");

        // Hoan tat va ket thuc mo phong
        #50;
        $finish;
    end

    // -----------------------------------------------------------
    // 8. Khoi bao ve chong treo (Timeout Watchdog)
    // -----------------------------------------------------------
    initial begin
        // Timeout 5ms de dam bao mang CNN du thoi gian tinh toan
        #5000000; 
        $display("\n[LOI] Mo phong bi treo! He thong vuot qua thoi gian 5ms (Timeout).\n");
        $finish;
    end

endmodule