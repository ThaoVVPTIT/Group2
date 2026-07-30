`timescale 1ns / 1ps

module tb_experiment;

    reg clk;
    reg aresetn;
    reg [7:0] s_tdata;
    reg s_tvalid;
    wire s_tready;
    wire [7:0] m_tdata;
    wire m_tvalid;

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

    always #5 clk = ~clk;

    reg [7:0] images [0:783999];
    reg [7:0] labels [0:999];

    initial begin
        $readmemh("input_1000.mem", images);
        $readmemh("label_1000.mem", labels);
    end

    integer img_idx, px_idx, r, c, raw_idx;
    integer mode;
    reg [7:0] pixel_val;

    initial begin
        clk = 0;
        aresetn = 0;
        s_tdata = 0;
        s_tvalid = 0;
        #100;

        for (mode = 0; mode < 4; mode = mode + 1) begin
            $display("--- TESTING MODE %0d ---", mode);
            for (img_idx = 0; img_idx < 10; img_idx = img_idx + 1) begin
                
                @(posedge clk);
                s_tvalid = 0;
                aresetn = 0;
                #50;
                aresetn = 1;
                #50;

                for (px_idx = 0; px_idx < 784; px_idx = px_idx + 1) begin
                    r = px_idx / 28;
                    c = px_idx % 28;
                    raw_idx = c * 28 + r; // Transposed
                    
                    if (mode == 0)      pixel_val = images[img_idx * 784 + px_idx]; // Raw
                    else if (mode == 1) pixel_val = images[img_idx * 784 + raw_idx]; // Transposed
                    else if (mode == 2) pixel_val = images[img_idx * 784 + px_idx] >> 1; // Raw / 2
                    else if (mode == 3) pixel_val = images[img_idx * 784 + raw_idx] >> 1; // Transposed / 2
                    
                    @(posedge clk);
                    s_tdata  = pixel_val;
                    s_tvalid = 1;
                    while (!s_tready) @(posedge clk);
                end
                
                @(posedge clk);
                s_tvalid = 0;

                wait (m_tvalid === 1'b1);
                @(negedge clk);
                
                $display("Mode %0d, Image %0d: Pred = %0d, Label = %0d", mode, img_idx, m_tdata, labels[img_idx]);
            end
        end
        $finish;
    end
endmodule
