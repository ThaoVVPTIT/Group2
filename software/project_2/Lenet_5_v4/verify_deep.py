"""
Deep verification: Load .keras model, compare weights with .mem files,
and test the model directly on input_1000.mem images.
"""
import numpy as np
import os
import sys

NEW_DIR = r"D:\THUC TAP\New folder"
SIM_DIR = r"D:\THUC TAP\Project_vivado\cnn_accelerator\cnn_accelerator.sim\sim_1\behav\xsim"

# ---- 1. Load Keras model and extract weights ----
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    model = tf.keras.models.load_model(os.path.join(NEW_DIR, "lenet5_emnist_3x3.keras"))
    print("Keras model loaded successfully!")
    model.summary()
    
    W_c1, b_c1 = model.get_layer("conv1").get_weights()
    W_c2, b_c2 = model.get_layer("conv2").get_weights()
    W_f1, b_f1 = model.get_layer("fc1").get_weights()
    W_f2, b_f2 = model.get_layer("fc2").get_weights()
    W_f3, b_f3 = model.get_layer("output").get_weights()
    
    print(f"\nKeras weights shapes:")
    print(f"  conv1: kernel={W_c1.shape}, bias={b_c1.shape}")
    print(f"  conv2: kernel={W_c2.shape}, bias={b_c2.shape}")
    print(f"  fc1:   kernel={W_f1.shape}, bias={b_f1.shape}")
    print(f"  fc2:   kernel={W_f2.shape}, bias={b_f2.shape}")
    print(f"  fc3:   kernel={W_f3.shape}, bias={b_f3.shape}")
    
except Exception as e:
    print(f"Could not load Keras model: {e}")
    sys.exit(1)

# ---- 2. Read input_1000.mem and labels ----
with open(os.path.join(NEW_DIR, "input_1000.mem"), 'r') as f:
    img_hex = [x.strip() for x in f.readlines() if x.strip()]
images_raw = np.array([int(x, 16) for x in img_hex], dtype=np.uint8)

with open(os.path.join(NEW_DIR, "label_1000.mem"), 'r') as f:
    label_hex = [x.strip() for x in f.readlines() if x.strip()]
labels = np.array([int(x, 16) for x in label_hex], dtype=np.int64)

print(f"\nLoaded {len(images_raw)//784} images, {len(labels)} labels")
print(f"Label range: {labels.min()} to {labels.max()}")
print(f"First 10 labels: {labels[:10]}")

# ---- 3. Test Keras float model directly on input_1000.mem images ----
print("\n=== Testing Keras float32 model on input_1000.mem ===")

# Strategy A: Raw uint8 / 255.0, no transpose
correct_a = 0
# Strategy B: Raw uint8 / 255.0, with transpose
correct_b = 0
# Strategy C: Transpose then raw
correct_c = 0

NUM_TEST = 100

for i in range(NUM_TEST):
    pixels = images_raw[i*784:(i+1)*784].astype(np.float32)
    img_28 = pixels.reshape(28, 28) / 255.0
    
    # Strategy A: no transpose
    pred_a = np.argmax(model.predict(img_28.reshape(1, 28, 28, 1), verbose=0))
    if pred_a == labels[i]:
        correct_a += 1
    
    # Strategy B: transpose
    img_t = img_28.T
    pred_b = np.argmax(model.predict(img_t.reshape(1, 28, 28, 1), verbose=0))
    if pred_b == labels[i]:
        correct_b += 1

print(f"  No transpose:   {correct_a}/{NUM_TEST} = {correct_a/NUM_TEST*100:.1f}%")
print(f"  With transpose: {correct_b}/{NUM_TEST} = {correct_b/NUM_TEST*100:.1f}%")

# ---- 4. Reproduce quantization from train.py and compare with .mem files ----
print("\n=== Reproducing quantization from train.py ===")

# Transpose weights like train.py does
W_c1_nchw = np.transpose(W_c1, (3, 2, 0, 1))  # (6, 1, 3, 3)
W_c2_nchw = np.transpose(W_c2, (3, 2, 0, 1))  # (16, 6, 3, 3)

W_f1_reshaped = np.reshape(W_f1, (5, 5, 16, 120))
W_f1_reshaped = np.transpose(W_f1_reshaped, (2, 0, 1, 3))
W_f1_reshaped = np.reshape(W_f1_reshaped, (400, 120))
W_f1_nchw = np.transpose(W_f1_reshaped, (1, 0))  # (120, 400)

W_f2_nchw = np.transpose(W_f2, (1, 0))  # (84, 120)
W_f3_nchw = np.transpose(W_f3, (1, 0))  # (47, 84)

# Calibrate scales (need to run model on calibration data)
# For now, use S_in_conv1 = 127.0 as in train.py
S_in_conv1 = 127.0

def per_channel_scale(W):
    axes = tuple(range(1, W.ndim))
    max_abs = np.maximum(np.max(np.abs(W), axis=axes), 1e-8)
    return 127.0 / max_abs

c1_w_scale = per_channel_scale(W_c1_nchw)
c2_w_scale = per_channel_scale(W_c2_nchw)
f1_w_scale = per_channel_scale(W_f1_nchw)
f2_w_scale = per_channel_scale(W_f2_nchw)

# Quantize kernel
def quantize_weights_pc(W, scale_pc):
    shape = [-1] + [1] * (W.ndim - 1)
    scale = scale_pc.reshape(shape)
    return np.clip(np.round(W * scale), -128, 127).astype(np.int64)

c1_w_int = quantize_weights_pc(W_c1_nchw, c1_w_scale)

# Read conv1_kernel.mem
def to_int8(val):
    val = val & 0xFF
    return val if val < 128 else val - 256

with open(os.path.join(SIM_DIR, "conv1_kernel.mem"), 'r') as f:
    mem_c1_kernel = np.array([to_int8(int(x.strip(), 16)) for x in f.readlines() if x.strip()], dtype=np.int64)

print(f"\nConv1 kernel comparison (first 9 values):")
print(f"  From .keras quantized: {c1_w_int.flatten()[:9]}")
print(f"  From .mem file:        {mem_c1_kernel[:9]}")
print(f"  Match: {np.array_equal(c1_w_int.flatten(), mem_c1_kernel)}")

if not np.array_equal(c1_w_int.flatten(), mem_c1_kernel):
    diff = np.abs(c1_w_int.flatten() - mem_c1_kernel)
    print(f"  Max difference: {diff.max()}")
    print(f"  Num different: {np.sum(diff > 0)}")
    # Show where they differ
    for idx in range(min(54, len(mem_c1_kernel))):
        if c1_w_int.flatten()[idx] != mem_c1_kernel[idx]:
            print(f"    [{idx}]: keras={c1_w_int.flatten()[idx]}, mem={mem_c1_kernel[idx]}")

# ---- 5. Check the calibrated output scales ----
# We need to run calibration to get S_out_conv1 etc.
# Let's use a few images from the test set
print("\n=== Running calibration to get output scales ===")

calib_model = tf.keras.Model(
    inputs=model.inputs,    
    outputs=[
        model.get_layer("conv1").output,
        model.get_layer("conv2").output,
        model.get_layer("fc1").output,
        model.get_layer("fc2").output,
    ],
)

# Use first 200 images from input_1000.mem as calibration
calib_imgs = []
for i in range(200):
    pixels = images_raw[i*784:(i+1)*784].astype(np.float32)
    img_28 = pixels.reshape(28, 28) / 255.0
    calib_imgs.append(img_28)
calib_imgs = np.array(calib_imgs).reshape(-1, 28, 28, 1)

conv1_act, conv2_act, fc1_act, fc2_act = calib_model.predict(calib_imgs, batch_size=128, verbose=0)

def calc_scale(activation):
    return 127.0 / max(np.max(np.abs(activation)), 1e-8)

S_out_conv1 = calc_scale(conv1_act)
S_out_conv2 = calc_scale(conv2_act)
S_out_fc1 = calc_scale(fc1_act)
S_out_fc2 = calc_scale(fc2_act)

print(f"Calibrated scales (from input_1000.mem images):")
print(f"  S_in_conv1 : {S_in_conv1:.4f}")
print(f"  S_out_conv1: {S_out_conv1:.4f}")
print(f"  S_out_conv2: {S_out_conv2:.4f}")
print(f"  S_out_fc1  : {S_out_fc1:.4f}")
print(f"  S_out_fc2  : {S_out_fc2:.4f}")

# Now compute multiplier/shift
def quantize_multiplier(real_multiplier, mantissa_bits=15):
    if real_multiplier <= 0:
        return 0, 0
    m = real_multiplier
    shift = 0
    while m >= 1.0:
        m /= 2.0
        shift -= 1
    while m < 0.5:
        m *= 2.0
        shift += 1
    mantissa = int(round(m * (1 << mantissa_bits)))
    if mantissa == (1 << mantissa_bits):
        mantissa //= 2
        shift -= 1
    total_shift = shift + mantissa_bits
    return mantissa, total_shift

def compute_mult_shift_array(S_in, S_out, w_scale_array):
    n = len(w_scale_array)
    mult = np.zeros(n, dtype=np.int64)
    shift = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        real_m = S_out / (S_in * w_scale_array[idx])
        m, s = quantize_multiplier(real_m)
        mult[idx] = m
        shift[idx] = s
    return mult, shift

c1_mult_py, c1_shift_py = compute_mult_shift_array(S_in_conv1, S_out_conv1, c1_w_scale)

# Read from .mem
with open(os.path.join(SIM_DIR, "conv1_multiplier.mem"), 'r') as f:
    mem_c1_mult = np.array([int(x.strip(), 16) for x in f.readlines() if x.strip()], dtype=np.int64)
with open(os.path.join(SIM_DIR, "conv1_shift.mem"), 'r') as f:
    mem_c1_shift = np.array([int(x.strip(), 16) for x in f.readlines() if x.strip()], dtype=np.int64)

print(f"\nConv1 multiplier comparison:")
print(f"  From .keras: {c1_mult_py}")
print(f"  From .mem:   {mem_c1_mult}")
print(f"  Match: {np.array_equal(c1_mult_py, mem_c1_mult)}")

print(f"\nConv1 shift comparison:")
print(f"  From .keras: {c1_shift_py}")
print(f"  From .mem:   {mem_c1_shift}")
print(f"  Match: {np.array_equal(c1_shift_py, mem_c1_shift)}")

# ---- 6. Run golden model using freshly computed weights ----
print("\n=== Running golden model with FRESH weights from .keras ===")

c1_b_int = np.round(b_c1 * S_in_conv1 * c1_w_scale).astype(np.int64)
c2_b_int = np.round(b_c2 * S_out_conv1 * c2_w_scale).astype(np.int64)
f1_b_int = np.round(b_f1 * S_out_conv2 * f1_w_scale).astype(np.int64)
f2_b_int = np.round(b_f2 * S_out_fc1 * f2_w_scale).astype(np.int64)
f3_s = 127.0 / max(np.max(np.abs(W_f3_nchw)), 1e-8)
f3_w_int = np.clip(np.round(W_f3_nchw * f3_s), -128, 127).astype(np.int64)
f3_b_int = np.round(b_f3 * S_out_fc2 * f3_s).astype(np.int64)

c2_w_int = quantize_weights_pc(W_c2_nchw, c2_w_scale)
f1_w_int = quantize_weights_pc(W_f1_nchw, f1_w_scale)
f2_w_int = quantize_weights_pc(W_f2_nchw, f2_w_scale)

c2_mult_py, c2_shift_py = compute_mult_shift_array(S_out_conv1, S_out_conv2, c2_w_scale)
f1_mult_py, f1_shift_py = compute_mult_shift_array(S_out_conv2, S_out_fc1, f1_w_scale)
f2_mult_py, f2_shift_py = compute_mult_shift_array(S_out_fc1, S_out_fc2, f2_w_scale)

def requantize(acc_val, mult_val, shift_val):
    product = int(acc_val) * int(mult_val)
    if shift_val > 0:
        product = product + (1 << (shift_val - 1))
        val = product >> shift_val
    else:
        val = product
    val = max(-128, min(127, val))
    val = max(0, val)  # ReLU
    return val

correct_fresh = 0
for i in range(NUM_TEST):
    pixels = images_raw[i*784:(i+1)*784].astype(np.float32)
    img_f = pixels.reshape(28, 28) / 255.0
    x = np.clip(np.round(img_f * S_in_conv1), -128, 127).astype(np.int64)
    
    # CONV1
    c1_out = np.zeros((6, 26, 26), dtype=np.int64)
    for oc in range(6):
        for r in range(26):
            for cc in range(26):
                acc = int(c1_b_int[oc])
                for ky in range(3):
                    for kx in range(3):
                        acc += int(x[r+ky, cc+kx]) * int(c1_w_int[oc, 0, ky, kx])
                c1_out[oc, r, cc] = requantize(acc, c1_mult_py[oc], c1_shift_py[oc])
    
    p1 = c1_out.reshape(6, 13, 2, 13, 2).max(axis=(2, 4))
    
    # CONV2
    c2_out = np.zeros((16, 11, 11), dtype=np.int64)
    for oc in range(16):
        for r in range(11):
            for cc in range(11):
                acc = int(c2_b_int[oc])
                for ic in range(6):
                    for ky in range(3):
                        for kx in range(3):
                            acc += int(p1[ic, r+ky, cc+kx]) * int(c2_w_int[oc, ic, ky, kx])
                c2_out[oc, r, cc] = requantize(acc, c2_mult_py[oc], c2_shift_py[oc])
    
    p2 = c2_out[:, :10, :10].reshape(16, 5, 2, 5, 2).max(axis=(2, 4))
    p2_flat = p2.reshape(400)
    
    fc1_out = np.zeros(120, dtype=np.int64)
    for n in range(120):
        acc = int(f1_b_int[n]) + int(np.dot(p2_flat, f1_w_int[n]))
        fc1_out[n] = requantize(acc, f1_mult_py[n], f1_shift_py[n])
    
    fc2_out = np.zeros(84, dtype=np.int64)
    for n in range(84):
        acc = int(f2_b_int[n]) + int(np.dot(fc1_out, f2_w_int[n]))
        fc2_out[n] = requantize(acc, f2_mult_py[n], f2_shift_py[n])
    
    fc3_out = np.zeros(47, dtype=np.int64)
    for n in range(47):
        acc = int(f3_b_int[n]) + int(np.dot(fc2_out, f3_w_int[n]))
        fc3_out[n] = acc
    
    pred = int(np.argmax(fc3_out))
    if pred == labels[i]:
        correct_fresh += 1

print(f"  Fresh weights + input/255*127: {correct_fresh}/{NUM_TEST} = {correct_fresh/NUM_TEST*100:.1f}%")

print("\nDone!")
