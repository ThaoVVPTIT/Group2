"""
Verify RTL vs Python golden model.
This script reads the .mem weight files used by RTL simulation,
reads the 1000 test images, and runs inference using the EXACT same
arithmetic as the Python golden model (simulate_rtl_int8_v2 in train.py).
Then it compares against the labels to find the expected accuracy.
"""
import numpy as np
import os

SIM_DIR = r"D:\THUC TAP\Project_vivado\cnn_accelerator\cnn_accelerator.sim\sim_1\behav\xsim"
NEW_DIR = r"D:\THUC TAP\New folder"

def to_int8(val):
    val = val & 0xFF
    return val if val < 128 else val - 256

def to_int32(val):
    val = val & 0xFFFFFFFF
    return val if val < 0x80000000 else val - 0x100000000

def read_hex_8(filepath):
    with open(filepath, 'r') as f:
        return np.array([to_int8(int(x.strip(), 16)) for x in f.readlines() if x.strip()], dtype=np.int64)

def read_hex_32(filepath):
    with open(filepath, 'r') as f:
        return np.array([to_int32(int(x.strip(), 16)) for x in f.readlines() if x.strip()], dtype=np.int64)

def read_hex_16u(filepath):
    with open(filepath, 'r') as f:
        return np.array([int(x.strip(), 16) for x in f.readlines() if x.strip()], dtype=np.int64)

def read_hex_8u(filepath):
    with open(filepath, 'r') as f:
        return np.array([int(x.strip(), 16) & 0xFF for x in f.readlines() if x.strip()], dtype=np.int64)

# ---- Read weights from the simulation directory ----
print("Reading weights from:", SIM_DIR)

c1_kernel = read_hex_8(os.path.join(SIM_DIR, "conv1_kernel.mem"))
c1_bias = read_hex_32(os.path.join(SIM_DIR, "conv1_bias.mem"))
c1_mult = read_hex_16u(os.path.join(SIM_DIR, "conv1_multiplier.mem"))
c1_shift = read_hex_8u(os.path.join(SIM_DIR, "conv1_shift.mem"))

c2_kernel = read_hex_8(os.path.join(SIM_DIR, "conv2_kernel.mem"))
c2_bias = read_hex_32(os.path.join(SIM_DIR, "conv2_bias.mem"))
c2_mult = read_hex_16u(os.path.join(SIM_DIR, "conv2_multiplier.mem"))
c2_shift = read_hex_8u(os.path.join(SIM_DIR, "conv2_shift.mem"))

f1_kernel = read_hex_8(os.path.join(SIM_DIR, "fc1_kernel.mem"))
f1_bias = read_hex_32(os.path.join(SIM_DIR, "fc1_bias.mem"))
f1_mult = read_hex_16u(os.path.join(SIM_DIR, "fc1_multiplier.mem"))
f1_shift = read_hex_8u(os.path.join(SIM_DIR, "fc1_shift.mem"))

f2_kernel = read_hex_8(os.path.join(SIM_DIR, "fc2_kernel.mem"))
f2_bias = read_hex_32(os.path.join(SIM_DIR, "fc2_bias.mem"))
f2_mult = read_hex_16u(os.path.join(SIM_DIR, "fc2_multiplier.mem"))
f2_shift = read_hex_8u(os.path.join(SIM_DIR, "fc2_shift.mem"))

f3_kernel = read_hex_8(os.path.join(SIM_DIR, "fc3_kernel.mem"))
f3_bias = read_hex_32(os.path.join(SIM_DIR, "fc3_bias.mem"))

print(f"Conv1 kernel: {len(c1_kernel)} values, bias: {len(c1_bias)}, mult: {len(c1_mult)}, shift: {len(c1_shift)}")
print(f"Conv2 kernel: {len(c2_kernel)} values, bias: {len(c2_bias)}, mult: {len(c2_mult)}, shift: {len(c2_shift)}")
print(f"FC1   kernel: {len(f1_kernel)} values, bias: {len(f1_bias)}, mult: {len(f1_mult)}, shift: {len(f1_shift)}")
print(f"FC2   kernel: {len(f2_kernel)} values, bias: {len(f2_bias)}, mult: {len(f2_mult)}, shift: {len(f2_shift)}")
print(f"FC3   kernel: {len(f3_kernel)} values, bias: {len(f3_bias)}")

# ---- Read test images and labels ----
print("\nReading test images and labels...")
with open(os.path.join(NEW_DIR, "input_1000.mem"), 'r') as f:
    img_hex = [x.strip() for x in f.readlines() if x.strip()]
images_raw = np.array([int(x, 16) for x in img_hex], dtype=np.int64)
print(f"Total image pixels: {len(images_raw)} (expect 784000)")

with open(os.path.join(NEW_DIR, "label_1000.mem"), 'r') as f:
    label_hex = [x.strip() for x in f.readlines() if x.strip()]
labels = np.array([int(x, 16) for x in label_hex], dtype=np.int64)
print(f"Total labels: {len(labels)} (expect 1000)")

# ---- Reshape weights ----
# Conv1: 6 output channels, 1 input channel, 3x3 kernel = 54 values
c1_w = c1_kernel.reshape(6, 1, 3, 3)
# Conv2: 16 output channels, 6 input channels, 3x3 kernel = 864 values
c2_w = c2_kernel.reshape(16, 6, 3, 3)
# FC1: 120 output, 400 input = 48000 values
f1_w = f1_kernel.reshape(120, 400)
# FC2: 84 output, 120 input = 10080 values
f2_w = f2_kernel.reshape(84, 120)
# FC3: 47 output, 84 input = 3948 values
f3_w = f3_kernel.reshape(47, 84)


# ---- Requantize function (matches train.py exactly) ----
def requantize_golden(acc_val, mult_val, shift_val):
    """Golden model requantize: multiply, shift, clip, ReLU"""
    product = int(acc_val) * int(mult_val)
    if shift_val > 0:
        product = product + (1 << (shift_val - 1))
        val = product >> shift_val
    else:
        val = product
    val = max(-128, min(127, val))
    val = max(0, val)  # ReLU
    return val

def requantize_rtl(acc_val, mult_val, shift_val):
    """RTL conv_calc.v requantize: multiply, shift, clip (NO ReLU)"""
    product = int(acc_val) * int(mult_val)
    if shift_val > 0:
        product = product + (1 << (shift_val - 1))
        val = product >> shift_val
    else:
        val = product
    val = max(-128, min(127, val))
    # No ReLU! RTL conv_calc does NOT apply ReLU
    return val

def requantize_rtl_fc(acc_val, mult_val, shift_val):
    """RTL fc1_seq/fc2_seq requantize: multiply, shift, offset -128, clip"""
    product = int(acc_val) * int(mult_val)
    if shift_val > 0:
        product = product + (1 << (shift_val - 1))
        val = product >> shift_val
    else:
        val = product
    val = val - 128  # RTL does acc = acc - 128
    val = max(-128, min(127, val))
    return val


def run_inference_golden(img_28x28, debug=False):
    """Run using the GOLDEN model logic from train.py"""
    x = img_28x28.astype(np.int64)

    # CONV1: 6 output channels, 3x3 kernel, valid padding -> 26x26
    c1_out = np.zeros((6, 26, 26), dtype=np.int64)
    for oc in range(6):
        for r in range(26):
            for c in range(26):
                acc = int(c1_bias[oc])
                for ky in range(3):
                    for kx in range(3):
                        acc += int(x[r+ky, c+kx]) * int(c1_w[oc, 0, ky, kx])
                c1_out[oc, r, c] = requantize_golden(acc, c1_mult[oc], c1_shift[oc])

    # POOL1: 2x2 max, 26x26 -> 13x13
    p1 = c1_out.reshape(6, 13, 2, 13, 2).max(axis=(2, 4))

    # CONV2: 16 output channels, 6 input, 3x3 kernel -> 11x11
    c2_out = np.zeros((16, 11, 11), dtype=np.int64)
    for oc in range(16):
        for r in range(11):
            for c in range(11):
                acc = int(c2_bias[oc])
                for ic in range(6):
                    for ky in range(3):
                        for kx in range(3):
                            acc += int(p1[ic, r+ky, c+kx]) * int(c2_w[oc, ic, ky, kx])
                c2_out[oc, r, c] = requantize_golden(acc, c2_mult[oc], c2_shift[oc])

    # POOL2: 2x2 max, 11x11 -> crop to 10x10 first, then 5x5
    p2 = c2_out[:, :10, :10].reshape(16, 5, 2, 5, 2).max(axis=(2, 4))
    p2_flat = p2.reshape(400)

    # FC1: 120 neurons
    fc1_out = np.zeros(120, dtype=np.int64)
    for n in range(120):
        acc = int(f1_bias[n])
        for i in range(400):
            acc += int(p2_flat[i]) * int(f1_w[n, i])
        fc1_out[n] = requantize_golden(acc, f1_mult[n], f1_shift[n])

    # FC2: 84 neurons
    fc2_out = np.zeros(84, dtype=np.int64)
    for n in range(84):
        acc = int(f2_bias[n])
        for i in range(120):
            acc += int(fc1_out[i]) * int(f2_w[n, i])
        fc2_out[n] = requantize_golden(acc, f2_mult[n], f2_shift[n])

    # FC3: 47 neurons, raw accumulator (no requant)
    fc3_out = np.zeros(47, dtype=np.int64)
    for n in range(47):
        acc = int(f3_bias[n])
        for i in range(84):
            acc += int(fc2_out[i]) * int(f3_w[n, i])
        fc3_out[n] = acc

    return int(np.argmax(fc3_out))


def run_inference_rtl(img_28x28):
    """Run using the ACTUAL RTL logic (conv_calc: no ReLU, fc: -128 offset)"""
    x = img_28x28.astype(np.int64)

    # CONV1: no ReLU in conv_calc
    c1_out = np.zeros((6, 26, 26), dtype=np.int64)
    for oc in range(6):
        for r in range(26):
            for c in range(26):
                acc = int(c1_bias[oc])
                for ky in range(3):
                    for kx in range(3):
                        acc += int(x[r+ky, c+kx]) * int(c1_w[oc, 0, ky, kx])
                c1_out[oc, r, c] = requantize_rtl(acc, c1_mult[oc], c1_shift[oc])

    # POOL1: max pooling (no separate ReLU in maxpool_relu module)
    p1 = c1_out.reshape(6, 13, 2, 13, 2).max(axis=(2, 4))

    # CONV2: no ReLU
    c2_out = np.zeros((16, 11, 11), dtype=np.int64)
    for oc in range(16):
        for r in range(11):
            for c in range(11):
                acc = int(c2_bias[oc])
                for ic in range(6):
                    for ky in range(3):
                        for kx in range(3):
                            acc += int(p1[ic, r+ky, c+kx]) * int(c2_w[oc, ic, ky, kx])
                c2_out[oc, r, c] = requantize_rtl(acc, c2_mult[oc], c2_shift[oc])

    # POOL2
    p2 = c2_out[:, :10, :10].reshape(16, 5, 2, 5, 2).max(axis=(2, 4))
    p2_flat = p2.reshape(400)

    # FC1: with -128 offset (RTL logic)
    fc1_out = np.zeros(120, dtype=np.int64)
    for n in range(120):
        acc = int(f1_bias[n])
        for i in range(400):
            acc += int(p2_flat[i]) * int(f1_w[n, i])
        fc1_out[n] = requantize_rtl_fc(acc, f1_mult[n], f1_shift[n])

    # FC2: with -128 offset (RTL logic)
    fc2_out = np.zeros(84, dtype=np.int64)
    for n in range(84):
        acc = int(f2_bias[n])
        for i in range(120):
            acc += int(fc1_out[i]) * int(f2_w[n, i])
        fc2_out[n] = requantize_rtl_fc(acc, f2_mult[n], f2_shift[n])

    # FC3: raw
    fc3_out = np.zeros(47, dtype=np.int64)
    for n in range(47):
        acc = int(f3_bias[n])
        for i in range(84):
            acc += int(fc2_out[i]) * int(f3_w[n, i])
        fc3_out[n] = acc

    return int(np.argmax(fc3_out))


# ---- Test with first 50 images using different input strategies ----
NUM_TEST = 50

strategies = {
    "RAW (unsigned 0-255)": lambda img: img,
    "Signed (- 128)": lambda img: img - 128,
    "Shift right 1": lambda img: img >> 1,
}

print("\n=== GOLDEN MODEL (with ReLU, no FC offset) ===")
for name, transform in strategies.items():
    correct = 0
    for i in range(NUM_TEST):
        pixels = images_raw[i*784:(i+1)*784]
        img_28 = transform(pixels).reshape(28, 28)
        pred = run_inference_golden(img_28)
        if pred == labels[i]:
            correct += 1
    print(f"  {name}: {correct}/{NUM_TEST} = {correct/NUM_TEST*100:.1f}%")

print("\n=== RTL MODEL (no ReLU in conv, -128 offset in FC) ===")
for name, transform in strategies.items():
    correct = 0
    for i in range(NUM_TEST):
        pixels = images_raw[i*784:(i+1)*784]
        img_28 = transform(pixels).reshape(28, 28)
        pred = run_inference_rtl(img_28)
        if pred == labels[i]:
            correct += 1
    print(f"  {name}: {correct}/{NUM_TEST} = {correct/NUM_TEST*100:.1f}%")

# Also test with transposed input
print("\n=== GOLDEN MODEL + TRANSPOSED INPUT ===")
for name, transform in strategies.items():
    correct = 0
    for i in range(NUM_TEST):
        pixels = images_raw[i*784:(i+1)*784]
        img_28 = transform(pixels).reshape(28, 28).T  # Transpose
        pred = run_inference_golden(img_28)
        if pred == labels[i]:
            correct += 1
    print(f"  {name} (T): {correct}/{NUM_TEST} = {correct/NUM_TEST*100:.1f}%")

print("\nDone!")
