	component AXI_DMA is
		port (
			clk_clk              : in  std_logic := 'X'; -- clk
			msgdma_0_csr_irq_irq : out std_logic;        -- irq
			reset_reset_n        : in  std_logic := 'X'  -- reset_n
		);
	end component AXI_DMA;

	u0 : component AXI_DMA
		port map (
			clk_clk              => CONNECTED_TO_clk_clk,              --              clk.clk
			msgdma_0_csr_irq_irq => CONNECTED_TO_msgdma_0_csr_irq_irq, -- msgdma_0_csr_irq.irq
			reset_reset_n        => CONNECTED_TO_reset_reset_n         --            reset.reset_n
		);

