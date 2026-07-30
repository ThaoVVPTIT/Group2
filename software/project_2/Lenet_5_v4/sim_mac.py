import numpy as np

def to_int8(val):
    val = val & 0xFF
    return val if val < 128 else val - 256

# Read conv1 weights
with open('D:/THUC TAP/Project_vivado/cnn_accelerator/cnn_accelerator.sim/sim_1/behav/xsim/conv1_kernel.mem', 'r') as f:
    w_hex = f.read().split()
w = [to_int8(int(x, 16)) for x in w_hex]

# Read conv1 bias
with open('D:/THUC TAP/Project_vivado/cnn_accelerator/cnn_accelerator.sim/sim_1/behav/xsim/conv1_bias.mem', 'r') as f:
    b_hex = f.read().split()
b = [to_int8(int(x, 16)) for x in b_hex]

# Read conv1 multiplier and shift
with open('D:/THUC TAP/Project_vivado/cnn_accelerator/cnn_accelerator.sim/sim_1/behav/xsim/conv1_multiplier.mem', 'r') as f:
    m_hex = f.read().split()
mult = [int(x, 16) for x in m_hex]
with open('D:/THUC TAP/Project_vivado/cnn_accelerator/cnn_accelerator.sim/sim_1/behav/xsim/conv1_shift.mem', 'r') as f:
    s_hex = f.read().split()
shift = [int(x, 16) for x in s_hex]

def quantize(acc, m, s):
    # exact logic from conv_calc.v
    # q = (acc * mult) >>> shift
    # if q > 127: 127 else if q < -128: -128 else q
    res = (acc * m) >> s
    if res > 127: return 127
    if res < -128: return -128
    return res

# Read test_image.mem
with open('D:/THUC TAP/New folder/test_image.mem', 'r') as f:
    img1_hex = f.read().split()
img1 = [int(x, 16) for x in img1_hex]

# Compute full Conv1 for channel 0
out_w = 28 - 3 + 1
out_h = 28 - 3 + 1
out_img = []

for r in range(out_h):
    for c in range(out_w):
        acc = b[0]
        for tr in range(3):
            for tc in range(3):
                idx = (r + tr) * 28 + (c + tc)
                val = img1[idx]
                acc += val * w[tr * 3 + tc]
        q_out = quantize(acc, mult[0], shift[0])
        out_img.append(q_out & 0xFF)

print(f"First valid output Conv1[0] = {out_img[0]}")
print(f"Second valid output Conv1[0] = {out_img[1]}")

