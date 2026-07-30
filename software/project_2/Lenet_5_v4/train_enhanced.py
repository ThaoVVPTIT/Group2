"""
Enhanced LeNet-5 training script for EMNIST Balanced.
Boosts accuracy via Data Augmentation, Cosine Decay LR Scheduler, and 40 Epochs.
Re-exports updated int8 weights & memory files directly to Vivado project.
"""
import os
import random
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

NEW_DIR = r"D:\THUC TAP\New folder"
OUTPUT_MODEL = os.path.join(NEW_DIR, "lenet5_emnist_3x3.keras")
MEM_DIR = os.path.join(NEW_DIR, "memory_files")

VIVADO_MEM_1 = r"D:\THUC TAP\Project_vivado\cnn_accelerator\cnn_accelerator.srcs\sources_1\imports\memory_files"
VIVADO_MEM_2 = r"D:\THUC TAP\Project_vivado\cnn_accelerator\cnn_accelerator.sim\sim_1\behav\xsim"
VIVADO_MEM_3 = r"D:\THUC TAP\Project_vivado\cnn_accelerator\cnn_accelerator.ip_user_files\mem_init_files"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

NUM_CLASSES = 47
BATCH_SIZE = 128
EPOCHS = 40

print("Loading EMNIST Balanced dataset...")
(train_raw, test_raw), dataset_info = tfds.load(
    "emnist/balanced", split=["train", "test"], as_supervised=True, with_info=True
)

def correct_emnist_orientation(image):
    return tf.transpose(image, perm=[1, 0, 2])

def preprocess_train(image, label):
    image = correct_emnist_orientation(image)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return image, label

def preprocess_test(image, label):
    image = correct_emnist_orientation(image)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    train_raw
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .shuffle(20000, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_raw
    .map(preprocess_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("Building enhanced LeNet-5 model...")
# Data Augmentation layer
data_augmentation = Sequential([
    layers.RandomRotation(0.06, fill_mode='constant', fill_value=0.0),
    layers.RandomTranslation(0.04, 0.04, fill_mode='constant', fill_value=0.0),
    layers.RandomZoom(0.04, fill_mode='constant', fill_value=0.0),
], name="augmentation")

model = Sequential([
    layers.Input(shape=(28, 28, 1)),
    data_augmentation,
    layers.Conv2D(6, (3, 3), strides=(1, 1), padding="valid", activation="relu", name="conv1"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1"),
    layers.Conv2D(16, (3, 3), strides=(1, 1), padding="valid", activation="relu", name="conv2"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool2"),
    layers.Flatten(name="flatten"),
    layers.Dense(120, activation="relu", name="fc1"),
    layers.Dense(84, activation="relu", name="fc2"),
    layers.Dense(NUM_CLASSES, activation="softmax", name="output"),
])

# Cosine Decay Learning Rate Scheduler
steps_per_epoch = len(train_ds)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.002,
    decay_steps=EPOCHS * steps_per_epoch,
    alpha=0.005
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint(filepath=OUTPUT_MODEL, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
]

print("\nStarting training for 40 epochs...")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Load best model
best_model = tf.keras.models.load_model(OUTPUT_MODEL)
loss, acc = best_model.evaluate(test_ds, verbose=0)
print(f"\n>>> Best Enhanced Model Test Accuracy: {acc*100:.2f}% <<<")

# ---- Quantization & Re-export ----
print("\nRunning post-training quantization calibration on 5,000 images...")

calib_images = []
for img, _ in train_raw.take(5000):
    img_corrected = correct_emnist_orientation(img)
    calib_images.append(img_corrected.numpy().astype(np.float32) / 255.0)
calib_images = np.stack(calib_images, axis=0)

calib_model = tf.keras.Model(
    inputs=best_model.inputs,
    outputs=[
        best_model.get_layer("conv1").output,
        best_model.get_layer("conv2").output,
        best_model.get_layer("fc1").output,
        best_model.get_layer("fc2").output,
    ],
)

conv1_act, conv2_act, fc1_act, fc2_act = calib_model.predict(calib_images, batch_size=256, verbose=0)

def calc_scale(act):
    return 127.0 / max(np.max(np.abs(act)), 1e-8)

S_in_conv1 = 127.0
S_out_conv1 = calc_scale(conv1_act)
S_out_conv2 = calc_scale(conv2_act)
S_out_fc1 = calc_scale(fc1_act)
S_out_fc2 = calc_scale(fc2_act)

W_c1, b_c1 = best_model.get_layer("conv1").get_weights()
W_c2, b_c2 = best_model.get_layer("conv2").get_weights()
W_f1, b_f1 = best_model.get_layer("fc1").get_weights()
W_f2, b_f2 = best_model.get_layer("fc2").get_weights()
W_f3, b_f3 = best_model.get_layer("output").get_weights()

W_c1 = np.transpose(W_c1, (3, 2, 0, 1))
W_c2 = np.transpose(W_c2, (3, 2, 0, 1))
W_f1 = np.reshape(W_f1, (5, 5, 16, 120))
W_f1 = np.transpose(W_f1, (2, 0, 1, 3))
W_f1 = np.reshape(W_f1, (400, 120))
W_f1 = np.transpose(W_f1, (1, 0))
W_f2 = np.transpose(W_f2, (1, 0))
W_f3 = np.transpose(W_f3, (1, 0))

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

def write_hex(filepath, values, n_bits):
    mask = (1 << n_bits) - 1
    hex_digits = (n_bits + 3) // 4
    lines = [f"{(int(v) & mask):0{hex_digits}x}" for v in np.asarray(values).flatten()]
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")

for output_dir in [MEM_DIR, VIVADO_MEM_1, VIVADO_MEM_2, VIVADO_MEM_3]:
    os.makedirs(output_dir, exist_ok=True)
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

print("\nSuccessfully updated all .mem weight files across all project directories!")

# Quick check on 1000 images golden int8
with open(os.path.join(NEW_DIR, "input_1000.mem"), 'r') as f:
    img_hex = [x.strip() for x in f.readlines() if x.strip()]
images_raw = np.array([int(x, 16) for x in img_hex], dtype=np.uint8)

with open(os.path.join(NEW_DIR, "label_1000.mem"), 'r') as f:
    label_hex = [x.strip() for x in f.readlines() if x.strip()]
labels = np.array([int(x, 16) for x in label_hex], dtype=np.int64)

def rq(acc, mult, shift):
    p = int(acc)*int(mult)
    if shift > 0:
        p = p + (1<<(shift-1))
        v = p >> shift
    else:
        v = p
    v = max(-128, min(127, v))
    v = max(0, v)
    return v

correct_count = 0
for i in range(1000):
    x = (images_raw[i*784:(i+1)*784].reshape(28,28).astype(np.int64) >> 1)
    
    o1 = np.zeros((6,26,26),dtype=np.int64)
    for oc in range(6):
        for r in range(26):
            for cc in range(26):
                a = int(c1_b_int[oc])
                for ky in range(3):
                    for kx in range(3):
                        a += int(x[r+ky,cc+kx])*int(c1_w_int[oc,0,ky,kx])
                o1[oc,r,cc] = rq(a,c1_mult[oc],c1_shift[oc])
    
    p1 = o1.reshape(6,13,2,13,2).max(axis=(2,4))
    
    o2 = np.zeros((16,11,11),dtype=np.int64)
    for oc in range(16):
        for r in range(11):
            for cc in range(11):
                a = int(c2_b_int[oc])
                for ic in range(6):
                    for ky in range(3):
                        for kx in range(3):
                            a += int(p1[ic,r+ky,cc+kx])*int(c2_w_int[oc,ic,ky,kx])
                o2[oc,r,cc] = rq(a,c2_mult[oc],c2_shift[oc])
    
    p2 = o2[:,:10,:10].reshape(16,5,2,5,2).max(axis=(2,4))
    pf = p2.reshape(400)
    
    o3 = np.zeros(120,dtype=np.int64)
    for n in range(120):
        a = int(f1_b_int[n]) + int(np.dot(pf,f1_w_int[n]))
        o3[n] = rq(a,f1_mult[n],f1_shift[n])
    
    o4 = np.zeros(84,dtype=np.int64)
    for n in range(84):
        a = int(f2_b_int[n]) + int(np.dot(o3,f2_w_int[n]))
        o4[n] = rq(a,f2_mult[n],f2_shift[n])
    
    o5 = np.zeros(47,dtype=np.int64)
    for n in range(47):
        a = int(f3_b_int[n]) + int(np.dot(o4,f3_w_int[n]))
        o5[n] = a
    
    pred = int(np.argmax(o5))
    if pred == labels[i]:
        correct_count += 1

print(f"\n==================================================")
print(f" NEW GOLDEN INT8 MODEL ACCURACY: {correct_count}/1000 ({correct_count/10.0:.1f}%)")
print(f"==================================================")
