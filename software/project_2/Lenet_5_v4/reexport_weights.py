"""
Re-export all weight .mem files from the .keras model.
Uses input_1000.mem images for calibration (since EMNIST TFDS may not be cached).
"""
import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

NEW_DIR = r"D:\THUC TAP\New folder"
MEM_DIR = os.path.join(NEW_DIR, "memory_files")
SIM_DIR = r"D:\THUC TAP\Project_vivado\cnn_accelerator\cnn_accelerator.sim\sim_1\behav\xsim"

# ---- 1. Load model ----
model = tf.keras.models.load_model(os.path.join(NEW_DIR, "lenet5_emnist_3x3.keras"))
print("Model loaded!")

W_c1, b_c1 = model.get_layer("conv1").get_weights()
W_c2, b_c2 = model.get_layer("conv2").get_weights()
W_f1, b_f1 = model.get_layer("fc1").get_weights()
W_f2, b_f2 = model.get_layer("fc2").get_weights()
W_f3, b_f3 = model.get_layer("output").get_weights()

# ---- 2. Transpose weights from Keras NHWC to Verilog NCHW ----
W_c1 = np.transpose(W_c1, (3, 2, 0, 1))  # (6, 1, 3, 3)
W_c2 = np.transpose(W_c2, (3, 2, 0, 1))  # (16, 6, 3, 3)

W_f1 = np.reshape(W_f1, (5, 5, 16, 120))
W_f1 = np.transpose(W_f1, (2, 0, 1, 3))
W_f1 = np.reshape(W_f1, (400, 120))
W_f1 = np.transpose(W_f1, (1, 0))  # (120, 400)

W_f2 = np.transpose(W_f2, (1, 0))  # (84, 120)
W_f3 = np.transpose(W_f3, (1, 0))  # (47, 84)

# ---- 3. Calibrate output scales ----
# Try to use EMNIST TFDS for calibration, fallback to input_1000.mem
try:
    import tensorflow_datasets as tfds
    
    def correct_emnist_orientation(image):
        return tf.transpose(image, perm=[1, 0, 2])
    
    def preprocess_image(image, label):
        image = correct_emnist_orientation(image)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    (train_raw, _), _ = tfds.load("emnist/balanced", split=["train", "test"],
                                    as_supervised=True, with_info=True)
    
    calib_images = []
    for image, _ in train_raw.take(1000):
        img, _ = preprocess_image(image, 0)
        calib_images.append(img.numpy())
    calib_images = np.stack(calib_images, axis=0)
    print(f"Using EMNIST TFDS calibration data: {calib_images.shape}")
    CALIB_SOURCE = "TFDS"

except Exception as e:
    print(f"TFDS not available ({e}), using input_1000.mem for calibration")
    with open(os.path.join(NEW_DIR, "input_1000.mem"), 'r') as f:
        img_hex = [x.strip() for x in f.readlines() if x.strip()]
    images_raw = np.array([int(x, 16) for x in img_hex], dtype=np.uint8)
    calib_images = images_raw.reshape(1000, 28, 28, 1).astype(np.float32) / 255.0
    print(f"Using input_1000.mem calibration data: {calib_images.shape}")
    CALIB_SOURCE = "input_1000"

calib_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer("conv1").output,
        model.get_layer("conv2").output,
        model.get_layer("fc1").output,
        model.get_layer("fc2").output,
    ],
)

conv1_act, conv2_act, fc1_act, fc2_act = calib_model.predict(
    calib_images, batch_size=128, verbose=0
)

def calc_scale(activation):
    return 127.0 / max(np.max(np.abs(activation)), 1e-8)

S_in_conv1 = 127.0
S_out_conv1 = calc_scale(conv1_act)
S_out_conv2 = calc_scale(conv2_act)
S_out_fc1 = calc_scale(fc1_act)
S_out_fc2 = calc_scale(fc2_act)

print(f"\nCalibrated scales (source={CALIB_SOURCE}):")
print(f"  S_in_conv1 : {S_in_conv1:.4f}")
print(f"  S_out_conv1: {S_out_conv1:.4f}")
print(f"  S_out_conv2: {S_out_conv2:.4f}")
print(f"  S_out_fc1  : {S_out_fc1:.4f}")
print(f"  S_out_fc2  : {S_out_fc2:.4f}")

# ---- 4. Quantize weights ----
def per_channel_scale(W):
    axes = tuple(range(1, W.ndim))
    max_abs = np.maximum(np.max(np.abs(W), axis=axes), 1e-8)
    return 127.0 / max_abs

def quantize_weights_pc(W, scale_pc):
    shape = [-1] + [1] * (W.ndim - 1)
    scale = scale_pc.reshape(shape)
    return np.clip(np.round(W * scale), -128, 127).astype(np.int64)

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

def compute_mult_shift(S_in, S_out, w_scale_array):
    n = len(w_scale_array)
    mult = np.zeros(n, dtype=np.int64)
    shift = np.zeros(n, dtype=np.int64)
    for i in range(n):
        real_m = S_out / (S_in * w_scale_array[i])
        m, s = quantize_multiplier(real_m)
        mult[i] = m
        shift[i] = s
    return mult, shift

c1_w_scale = per_channel_scale(W_c1)
c2_w_scale = per_channel_scale(W_c2)
f1_w_scale = per_channel_scale(W_f1)
f2_w_scale = per_channel_scale(W_f2)
f3_s = 127.0 / max(np.max(np.abs(W_f3)), 1e-8)

c1_w_int = quantize_weights_pc(W_c1, c1_w_scale)
c2_w_int = quantize_weights_pc(W_c2, c2_w_scale)
f1_w_int = quantize_weights_pc(W_f1, f1_w_scale)
f2_w_int = quantize_weights_pc(W_f2, f2_w_scale)
f3_w_int = np.clip(np.round(W_f3 * f3_s), -128, 127).astype(np.int64)

c1_b_int = np.round(b_c1 * S_in_conv1 * c1_w_scale).astype(np.int64)
c2_b_int = np.round(b_c2 * S_out_conv1 * c2_w_scale).astype(np.int64)
f1_b_int = np.round(b_f1 * S_out_conv2 * f1_w_scale).astype(np.int64)
f2_b_int = np.round(b_f2 * S_out_fc1 * f2_w_scale).astype(np.int64)
f3_b_int = np.round(b_f3 * S_out_fc2 * f3_s).astype(np.int64)

c1_mult, c1_shift = compute_mult_shift(S_in_conv1, S_out_conv1, c1_w_scale)
c2_mult, c2_shift = compute_mult_shift(S_out_conv1, S_out_conv2, c2_w_scale)
f1_mult, f1_shift = compute_mult_shift(S_out_conv2, S_out_fc1, f1_w_scale)
f2_mult, f2_shift = compute_mult_shift(S_out_fc1, S_out_fc2, f2_w_scale)

# ---- 5. Write .mem files ----
def write_hex(filepath, values, n_bits):
    mask = (1 << n_bits) - 1
    hex_digits = (n_bits + 3) // 4
    lines = [f"{(int(v) & mask):0{hex_digits}x}" for v in np.asarray(values).flatten()]
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")

for output_dir in [MEM_DIR, SIM_DIR]:
    write_hex(os.path.join(output_dir, "conv1_kernel.mem"), c1_w_int, 8)
    write_hex(os.path.join(output_dir, "conv1_bias.mem"), c1_b_int, 32)
    write_hex(os.path.join(output_dir, "conv1_multiplier.mem"), c1_mult, 16)
    write_hex(os.path.join(output_dir, "conv1_shift.mem"), c1_shift, 8)

    write_hex(os.path.join(output_dir, "conv2_kernel.mem"), c2_w_int, 8)
    write_hex(os.path.join(output_dir, "conv2_bias.mem"), c2_b_int, 32)
    write_hex(os.path.join(output_dir, "conv2_multiplier.mem"), c2_mult, 16)
    write_hex(os.path.join(output_dir, "conv2_shift.mem"), c2_shift, 8)

    write_hex(os.path.join(output_dir, "fc1_kernel.mem"), f1_w_int, 8)
    write_hex(os.path.join(output_dir, "fc1_bias.mem"), f1_b_int, 32)
    write_hex(os.path.join(output_dir, "fc1_multiplier.mem"), f1_mult, 16)
    write_hex(os.path.join(output_dir, "fc1_shift.mem"), f1_shift, 8)

    write_hex(os.path.join(output_dir, "fc2_kernel.mem"), f2_w_int, 8)
    write_hex(os.path.join(output_dir, "fc2_bias.mem"), f2_b_int, 32)
    write_hex(os.path.join(output_dir, "fc2_multiplier.mem"), f2_mult, 16)
    write_hex(os.path.join(output_dir, "fc2_shift.mem"), f2_shift, 8)

    write_hex(os.path.join(output_dir, "fc3_kernel.mem"), f3_w_int, 8)
    write_hex(os.path.join(output_dir, "fc3_bias.mem"), f3_b_int, 32)

    print(f"Written all .mem files to: {output_dir}")

# ---- 6. Quick golden model verification ----
print("\n=== Quick golden model verification (50 images) ===")

with open(os.path.join(NEW_DIR, "input_1000.mem"), 'r') as f:
    img_hex = [x.strip() for x in f.readlines() if x.strip()]
images_raw = np.array([int(x, 16) for x in img_hex], dtype=np.uint8)

with open(os.path.join(NEW_DIR, "label_1000.mem"), 'r') as f:
    label_hex = [x.strip() for x in f.readlines() if x.strip()]
labels = np.array([int(x, 16) for x in label_hex], dtype=np.int64)

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

correct = 0
NUM_TEST = 50
for i in range(NUM_TEST):
    pixels = images_raw[i*784:(i+1)*784].astype(np.float32)
    img_f = pixels.reshape(28, 28) / 255.0
    x = np.clip(np.round(img_f * S_in_conv1), -128, 127).astype(np.int64)
    
    # Conv1
    c1 = np.zeros((6, 26, 26), dtype=np.int64)
    for oc in range(6):
        for r in range(26):
            for cc in range(26):
                acc = int(c1_b_int[oc])
                for ky in range(3):
                    for kx in range(3):
                        acc += int(x[r+ky, cc+kx]) * int(c1_w_int[oc, 0, ky, kx])
                c1[oc, r, cc] = requantize(acc, c1_mult[oc], c1_shift[oc])
    
    p1 = c1.reshape(6, 13, 2, 13, 2).max(axis=(2, 4))
    
    # Conv2
    c2 = np.zeros((16, 11, 11), dtype=np.int64)
    for oc in range(16):
        for r in range(11):
            for cc in range(11):
                acc = int(c2_b_int[oc])
                for ic in range(6):
                    for ky in range(3):
                        for kx in range(3):
                            acc += int(p1[ic, r+ky, cc+kx]) * int(c2_w_int[oc, ic, ky, kx])
                c2[oc, r, cc] = requantize(acc, c2_mult[oc], c2_shift[oc])
    
    p2 = c2[:, :10, :10].reshape(16, 5, 2, 5, 2).max(axis=(2, 4))
    p2_flat = p2.reshape(400)
    
    # FC1
    fc1 = np.zeros(120, dtype=np.int64)
    for n in range(120):
        acc = int(f1_b_int[n]) + int(np.dot(p2_flat, f1_w_int[n]))
        fc1[n] = requantize(acc, f1_mult[n], f1_shift[n])
    
    # FC2
    fc2 = np.zeros(84, dtype=np.int64)
    for n in range(84):
        acc = int(f2_b_int[n]) + int(np.dot(fc1, f2_w_int[n]))
        fc2[n] = requantize(acc, f2_mult[n], f2_shift[n])
    
    # FC3
    fc3 = np.zeros(47, dtype=np.int64)
    for n in range(47):
        acc = int(f3_b_int[n]) + int(np.dot(fc2, f3_w_int[n]))
        fc3[n] = acc
    
    pred = int(np.argmax(fc3))
    if pred == labels[i]:
        correct += 1
    if i < 10:
        print(f"  Image {i}: pred={pred}, label={labels[i]}, {'OK' if pred==labels[i] else 'WRONG'}")

print(f"\nGolden model accuracy: {correct}/{NUM_TEST} = {correct/NUM_TEST*100:.1f}%")
print("\nDone! All weight files have been re-exported.")
