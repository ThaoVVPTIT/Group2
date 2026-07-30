import os
import random


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.metrics import confusion_matrix, classification_report




# ============================================================
# 1. CẤU HÌNH
# ============================================================


SEED = 42
BATCH_SIZE = 128
EPOCHS = 20


NUM_CLASSES = 47
INPUT_SHAPE = (28, 28, 1)


# EMNIST Balanced:
# 10 chữ số + 37 lớp chữ cái đã gộp các ký tự hoa/thường dễ nhầm = 47 lớp
DATASET_NAME = "emnist/balanced"


DIGITS_ONLY_FOR_FPGA = False  # True => chỉ giữ 10 lớp digit khi export cho RTL


OUTPUT_DIR = "lenet5_emnist_3x3_output"
MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "lenet5_emnist_3x3.keras",
)
BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_lenet5_emnist_3x3.keras",
)


FPGA_EXPORT_DIR = os.path.join(OUTPUT_DIR, "fpga_export")


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FPGA_EXPORT_DIR, exist_ok=True)


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



print("TensorFlow version:", tf.__version__)
print("TensorFlow Datasets version:", tfds.__version__)




# ============================================================
# 2. ĐỌC BỘ DỮ LIỆU EMNIST BALANCED
# ============================================================


(train_raw, test_raw), dataset_info = tfds.load(
    DATASET_NAME,
    split=["train", "test"],
    as_supervised=True,
    with_info=True,
    shuffle_files=True,
)


train_size = dataset_info.splits["train"].num_examples
test_size = dataset_info.splits["test"].num_examples


BALANCED_ASCII_CODES = (
    list(range(ord("0"), ord("9") + 1))
    + list(range(ord("A"), ord("Z") + 1))
    + [
        ord("a"),
        ord("b"),
        ord("d"),
        ord("e"),
        ord("f"),
        ord("g"),
        ord("h"),
        ord("n"),
        ord("q"),
        ord("r"),
        ord("t"),
    ]
)


class_names = [
    chr(ascii_code)
    for ascii_code in BALANCED_ASCII_CODES
]


if len(class_names) != NUM_CLASSES:
    raise RuntimeError(
        f"Số tên lớp nhận được là {len(class_names)}, "
        f"nhưng mô hình yêu cầu {NUM_CLASSES} lớp."
    )


print("\nThông tin EMNIST Balanced:")
print("Số ảnh train:", train_size)
print("Số ảnh test :", test_size)
print("Số lớp      :", len(class_names))
print("Các lớp     :", class_names)


# ============================================================
# 3. HÀM TIỀN XỬ LÝ
# ============================================================


def correct_emnist_orientation(image):
    """
    Sửa hướng ảnh EMNIST từ TFDS.


    Input : (28, 28, 1)
    Output: (28, 28, 1)
    """
    return tf.transpose(image, perm=[1, 0, 2])




def preprocess_image(image, label):
    """
    - Sửa hướng ảnh.
    - Chuyển uint8 sang float32.
    - Chuẩn hóa pixel từ [0, 255] về [0, 1].
    - Giữ nhãn dạng số nguyên để dùng sparse loss.
    """
    image = correct_emnist_orientation(image)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)


    return image, label




AUTOTUNE = tf.data.AUTOTUNE


train_dataset = (
    train_raw
    .map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )
    .shuffle(
        buffer_size=20000,
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


test_dataset = (
    test_raw
    .map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)




# ============================================================
# 4. HIỂN THỊ 8 ẢNH NGẪU NHIÊN
# ============================================================


sample_dataset = (
    train_raw
    .shuffle(
        buffer_size=10000,
        seed=SEED,
        reshuffle_each_iteration=False,
    )
    .take(8)
)


sample_images = []
sample_labels = []


for image, label in tfds.as_numpy(sample_dataset):
    # Sửa hướng ảnh bằng transpose tương đương với hàm TensorFlow ở trên.
    image = np.transpose(image, (1, 0, 2))
    sample_images.append(image)
    sample_labels.append(int(label))


plt.figure(figsize=(10, 5))


for index, (image, label) in enumerate(
    zip(sample_images, sample_labels)
):
    plt.subplot(2, 4, index + 1)
    plt.imshow(image[:, :, 0], cmap="gray")
    plt.title(
        f"Nhãn {label}: {class_names[label]}"
    )
    plt.axis("off")


plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "random_emnist_balanced_images.png",
    ),
    dpi=300,
    bbox_inches="tight",
)
plt.show()




# ============================================================
# 5. TÍNH VÀ HIỂN THỊ PHÂN BỐ NHÃN
# ============================================================


label_counts = np.zeros(NUM_CLASSES, dtype=np.int64)


print("\nĐang đếm phân bố nhãn của tập train...")


for _, label_batch in train_raw.batch(4096):
    labels_numpy = label_batch.numpy()
    label_counts += np.bincount(
        labels_numpy,
        minlength=NUM_CLASSES,
    )


plt.figure(figsize=(18, 6))
positions = np.arange(NUM_CLASSES)


plt.bar(positions, label_counts)
plt.xticks(
    positions,
    [
        f"{index}\n{class_names[index]}"
        for index in range(NUM_CLASSES)
    ],
    fontsize=7,
)
plt.title("Phân bố 47 lớp trong EMNIST Balanced")
plt.xlabel("Chỉ số lớp và ký tự")
plt.ylabel("Số lượng ảnh")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "label_distribution.png",
    ),
    dpi=300,
    bbox_inches="tight",
)
plt.show()




# ============================================================
# 6. XÂY DỰNG MÔ HÌNH LENET-5
# ============================================================


def create_lenet5():
    """
    LeNet-5 điều chỉnh cho EMNIST Balanced 28x28, dùng kernel 3x3.\n    Kích thước: 28 -> Conv1 26 -> Pool1 13 -> Conv2 11 -> Pool2 5.\n    Flatten = 16 x 5 x 5 = 400.
    """


    model = Sequential([
        Input(shape=INPUT_SHAPE),


        Conv2D(
            filters=6,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
            name="conv1",
        ),


        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name="pool1",
        ),


        Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
            name="conv2",
        ),


        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name="pool2",
        ),


        Flatten(name="flatten"),


        Dense(
            units=120,
            activation="relu",
            name="fc1",
        ),


        Dense(
            units=84,
            activation="relu",
            name="fc2",
        ),


        Dense(
            units=NUM_CLASSES,
            activation="softmax",
            name="output",
        ),
    ])


    return model




model = create_lenet5()
model.summary()




# ============================================================
# 7. BIÊN DỊCH MÔ HÌNH
# ============================================================


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)




# ============================================================
# 8. CALLBACK
# ============================================================


callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),


    ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),


    ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
]




# ============================================================
# 9. HUẤN LUYỆN MÔ HÌNH
# ============================================================


train_fit_raw, validation_raw = tfds.load(
    DATASET_NAME,
    split=["train[:90%]", "train[90%:]"],
    as_supervised=True,
    shuffle_files=True,
)


train_fit_dataset = (
    train_fit_raw
    .map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )
    .shuffle(
        buffer_size=20000,
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


validation_dataset = (
    validation_raw
    .map(
        preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


history = model.fit(
    train_fit_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=callbacks,
    verbose=1,
)




# ============================================================
# 10. LƯU MÔ HÌNH
# ============================================================


model.save(MODEL_PATH)


print("\nĐã lưu mô hình cuối tại:", MODEL_PATH)
print("Đã lưu mô hình tốt nhất tại:", BEST_MODEL_PATH)




# ============================================================
# 11. ĐÁNH GIÁ TRÊN TẬP KIỂM TRA
# ============================================================


test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=2,
)


print(f"Test loss    : {test_loss:.6f}")
print(f"Test accuracy: {test_accuracy:.6f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")




# ============================================================
# 22. LƯỢNG TỬ HÓA THAM KHẢO BẰNG TFLITE (ĐỐI CHỨNG, KHÔNG DÙNG CHO RTL)
# ============================================================
# Mục này CHỈ để có một con số đối chứng "TFLite làm đúng thì được bao
# nhiêu %" - không phải nguồn trọng số xuất cho RTL. Trọng số thật cho
# RTL được tính riêng, thủ công, ở mục 23 bên dưới (per-channel).


def representative_dataset_gen():
    """
    Bộ dữ liệu đại diện dùng để hiệu chỉnh (calibrate) TFLite quantization.
    Lấy khoảng 200 ảnh từ tập train để ước lượng min/max activation
    của từng lớp.
    """
    calibration_dataset = (
        train_raw
        .take(200)
        .map(preprocess_image)
        .batch(1)
    )
    for image, _ in calibration_dataset:
        yield [image]




print("\nĐang lượng tử hóa mô hình về INT8 bằng TFLite (đối chứng)...")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


tflite_int8_model = converter.convert()


TFLITE_INT8_PATH = os.path.join(
    FPGA_EXPORT_DIR,
    "lenet5_emnist_3x3_int8.tflite",
)


with open(TFLITE_INT8_PATH, "wb") as file:
    file.write(tflite_int8_model)


print("Đã lưu mô hình INT8 tham khảo (.tflite) tại:", TFLITE_INT8_PATH)


interpreter = tf.lite.Interpreter(model_content=tflite_int8_model)
interpreter.allocate_tensors()


quant_report_lines = ["layer_name\tscale\tzero_point\tdtype"]
for tensor_detail in interpreter.get_tensor_details():
    quant_params = tensor_detail.get("quantization_parameters", {})
    scales = quant_params.get("scales", [])
    zero_points = quant_params.get("zero_points", [])
    scale_value = float(scales[0]) if len(scales) > 0 else 0.0
    zero_point_value = int(zero_points[0]) if len(zero_points) > 0 else 0
    quant_report_lines.append(
        f"{tensor_detail['name']}\t{scale_value:.10f}\t"
        f"{zero_point_value}\t{tensor_detail['dtype']}"
    )


QUANT_REPORT_PATH = os.path.join(
    FPGA_EXPORT_DIR,
    "quantization_scale_report.tsv",
)
with open(QUANT_REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("\n".join(quant_report_lines))


print("Đã lưu báo cáo scale/zero-point (TFLite) tại:", QUANT_REPORT_PATH)




def evaluate_tflite_int8(tflite_model_bytes, eval_dataset, num_samples=2000):
    interp = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interp.allocate_tensors()
    input_detail = interp.get_input_details()[0]
    output_detail = interp.get_output_details()[0]
    in_scale, in_zero_point = input_detail["quantization"]


    correct = 0
    total = 0


    for image_batch, label_batch in eval_dataset.unbatch().batch(1).take(num_samples):
        image_np = image_batch.numpy()
        image_int8 = np.round(image_np / in_scale + in_zero_point).astype(np.int8)
        interp.set_tensor(input_detail["index"], image_int8)
        interp.invoke()
        output_int8 = interp.get_tensor(output_detail["index"])
        predicted = int(np.argmax(output_int8[0]))
        true_label = int(label_batch.numpy()[0])
        correct += int(predicted == true_label)
        total += 1


    return correct / total if total > 0 else 0.0




tflite_int8_accuracy = evaluate_tflite_int8(tflite_int8_model, test_dataset)
print(f"Độ chính xác TFLite INT8 (đối chứng, mẫu con) : {tflite_int8_accuracy * 100:.2f}%")
print(f"Độ chính xác float32 (mục 11 ở trên)          : {test_accuracy * 100:.2f}%")


with open(QUANT_REPORT_PATH, "a", encoding="utf-8") as file:
    file.write(f"\n\n# Accuracy float32       : {test_accuracy:.6f}")
    file.write(f"\n# Accuracy TFLite int8 (đối chứng): {tflite_int8_accuracy:.6f}")




# ============================================================
# 22b. CALIBRATE SCALE ĐẦU RA THỰC TẾ CỦA TỪNG LAYER (post-ReLU)
# ============================================================
# Đây là bước THAY THẾ cho "shift cố định 8-bit" của bản cũ.
# Dùng activation thật (không suy luận từ công thức shift) để xác định
# scale hợp lý nhất cho từng layer - giống hệt cách TFLite converter
# dùng representative dataset để calibrate.


CALIB_NUM_IMAGES = 1000


calib_model = tf.keras.Model(
    inputs=model.inputs,    
    outputs=[
        model.get_layer("conv1").output,  # đã có ReLU built-in
        model.get_layer("conv2").output,
        model.get_layer("fc1").output,
        model.get_layer("fc2").output,
    ],
)


calib_images = []
for image, _ in train_raw.take(CALIB_NUM_IMAGES):
    img, _ = preprocess_image(image, 0)
    calib_images.append(img.numpy())
calib_images = np.stack(calib_images, axis=0)


conv1_act, conv2_act, fc1_act, fc2_act = calib_model.predict(
    calib_images, batch_size=128, verbose=0
)




def calc_scale(activation):
    """Scale = 127 / max(abs(activation)), tránh chia 0."""
    return 127.0 / max(np.max(np.abs(activation)), 1e-8)




S_in_conv1 = 127.0  # ảnh input đã chuẩn hóa [0,1], quy ước scale=127, không zero-center
S_out_conv1 = calc_scale(conv1_act)
S_out_conv2 = calc_scale(conv2_act)
S_out_fc1 = calc_scale(fc1_act)
S_out_fc2 = calc_scale(fc2_act)


print("\n--- SCALE CALIBRATE TỪNG LAYER (post-ReLU, thay cho shift=8 cố định) ---")
print(f"S_in_conv1 : {S_in_conv1:.4f}")
print(f"S_out_conv1: {S_out_conv1:.4f}")
print(f"S_out_conv2: {S_out_conv2:.4f}")
print(f"S_out_fc1  : {S_out_fc1:.4f}")
print(f"S_out_fc2  : {S_out_fc2:.4f}")




# ============================================================
# 23. LƯỢNG TỬ HÓA PER-CHANNEL (WEIGHT) + MULTIPLY-SHIFT (REQUANT)
# ============================================================
# Đây là nguồn trọng số THẬT xuất cho RTL. Mỗi kênh output của
# conv1/conv2/fc1/fc2 có scale trọng số riêng + multiplier/shift riêng
# (kiểu TFLite per-channel quantization). FC3 (lớp cuối) giữ per-tensor
# vì output của nó không bị lượng tử hóa lại - argmax dùng trực tiếp
# giá trị accumulator int32/int64 thô.


print("\n--- ĐANG XUẤT TRỌNG SỐ PER-CHANNEL CHO RTL ---")
WEIGHTS_EXPORT_DIR = os.path.join(FPGA_EXPORT_DIR, "weights_hex")
os.makedirs(WEIGHTS_EXPORT_DIR, exist_ok=True)




def quantize_multiplier(real_multiplier, mantissa_bits=15):
    """
    Phân rã real_multiplier (>0) thành:
        real_multiplier ~= mantissa / 2^shift
    mantissa là số nguyên mantissa_bits-bit, nằm trong [2^(bits-1), 2^bits - 1]
    (giống TFLite's QuantizeMultiplier, dùng cho requant per-channel int8).
    """
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




W_c1, b_c1 = model.get_layer("conv1").get_weights()
W_c2, b_c2 = model.get_layer("conv2").get_weights()
W_f1, b_f1 = model.get_layer("fc1").get_weights()
W_f2, b_f2 = model.get_layer("fc2").get_weights()
W_f3, b_f3 = model.get_layer("output").get_weights()


# 1. CHUYỂN VỊ MA TRẬN TỪ KERAS (NHWC) SANG VERILOG (NCHW)
W_c1 = np.transpose(W_c1, (3, 2, 0, 1))  # (6, 1, 3, 3)
W_c2 = np.transpose(W_c2, (3, 2, 0, 1))  # (16, 6, 3, 3)


# Lớp FC1: Khôi phục HWC -> Chuyển thành CHW -> Dẹt lại 400
W_f1 = np.reshape(W_f1, (5, 5, 16, 120))
W_f1 = np.transpose(W_f1, (2, 0, 1, 3))
W_f1 = np.reshape(W_f1, (400, 120))
W_f1 = np.transpose(W_f1, (1, 0))  # (120, 400)


W_f2 = np.transpose(W_f2, (1, 0))  # (84, 120)
W_f3 = np.transpose(W_f3, (1, 0))  # (47, 84)




def per_channel_scale(W):
    """Scale riêng cho từng kênh output = axis 0 của W (đã transpose)."""
    axes = tuple(range(1, W.ndim))
    max_abs = np.maximum(np.max(np.abs(W), axis=axes), 1e-8)
    return 127.0 / max_abs




def quantize_weights_per_channel(W, scale_per_channel):
    shape = [-1] + [1] * (W.ndim - 1)
    scale = scale_per_channel.reshape(shape)
    return np.clip(np.round(W * scale), -128, 127).astype(np.int64)




# ---- Per-channel weight scale ----
c1_w_scale = per_channel_scale(W_c1)  # (6,)
c2_w_scale = per_channel_scale(W_c2)  # (16,)
f1_w_scale = per_channel_scale(W_f1)  # (120,)
f2_w_scale = per_channel_scale(W_f2)  # (84,)
f3_s = 127.0 / max(np.max(np.abs(W_f3)), 1e-8)  # FC3: per-tensor (giữ nguyên)


# ---- Quantize weights ----
c1_w_int = quantize_weights_per_channel(W_c1, c1_w_scale)
c2_w_int = quantize_weights_per_channel(W_c2, c2_w_scale)
f1_w_int = quantize_weights_per_channel(W_f1, f1_w_scale)
f2_w_int = quantize_weights_per_channel(W_f2, f2_w_scale)
f3_w_int = np.clip(np.round(W_f3 * f3_s), -128, 127).astype(np.int64)


# ---- Quantize bias: scale = S_in_layer * S_w[c] (per-channel) ----
# KHÔNG cộng ROUND_OFFSET nữa - việc làm tròn giờ nằm ở bước requant
# (multiply-then-shift), không phải ở bias như bản per-tensor cũ.
c1_b_int = np.round(b_c1 * S_in_conv1 * c1_w_scale).astype(np.int64)
c2_b_int = np.round(b_c2 * S_out_conv1 * c2_w_scale).astype(np.int64)
f1_b_int = np.round(b_f1 * S_out_conv2 * f1_w_scale).astype(np.int64)
f2_b_int = np.round(b_f2 * S_out_fc1 * f2_w_scale).astype(np.int64)
f3_b_int = np.round(b_f3 * S_out_fc2 * f3_s).astype(np.int64)  # FC3: input scale = S_out_fc2


# ---- Multiplier + shift per-channel (requant) ----
def compute_mult_shift_array(S_in_layer, S_out_layer, w_scale_array):
    n = len(w_scale_array)
    mult = np.zeros(n, dtype=np.int64)
    shift = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        real_m = S_out_layer / (S_in_layer * w_scale_array[idx])
        m, s = quantize_multiplier(real_m)
        mult[idx] = m
        shift[idx] = s
    return mult, shift




c1_mult, c1_shift = compute_mult_shift_array(S_in_conv1, S_out_conv1, c1_w_scale)
c2_mult, c2_shift = compute_mult_shift_array(S_out_conv1, S_out_conv2, c2_w_scale)
f1_mult, f1_shift = compute_mult_shift_array(S_out_conv2, S_out_fc1, f1_w_scale)
f2_mult, f2_shift = compute_mult_shift_array(S_out_fc1, S_out_fc2, f2_w_scale)
# FC3: không cần multiplier/shift - giữ raw accumulator cho argmax


print("\nVí dụ multiplier/shift kênh 0 mỗi layer (kiểm tra sơ bộ):")
print(f"conv1[0]: mult={c1_mult[0]}, shift={c1_shift[0]}")
print(f"conv2[0]: mult={c2_mult[0]}, shift={c2_shift[0]}")
print(f"fc1[0]  : mult={f1_mult[0]}, shift={f1_shift[0]}")
print(f"fc2[0]  : mult={f2_mult[0]}, shift={f2_shift[0]}")


print("\nPhạm vi shift từng layer (để chọn độ rộng thanh ghi RTL cho an toàn):")
print(f"conv1 shift: min={c1_shift.min()}, max={c1_shift.max()}")
print(f"conv2 shift: min={c2_shift.min()}, max={c2_shift.max()}")
print(f"fc1   shift: min={f1_shift.min()}, max={f1_shift.max()}")
print(f"fc2   shift: min={f2_shift.min()}, max={f2_shift.max()}")




# ---- Ghi file hex ----
def write_hex(name, values, n_bits):
    mask = (1 << n_bits) - 1
    hex_digits = (n_bits + 3) // 4
    lines = [f"{(int(v) & mask):0{hex_digits}x}" for v in np.asarray(values).flatten()]
    with open(os.path.join(WEIGHTS_EXPORT_DIR, name), "w") as f:
        f.write("\n".join(lines) + "\n")
   


# Trọng số (8-bit) - giữ nguyên format cũ
write_hex("conv1_kernel.hex", c1_w_int, 8)
write_hex("conv2_kernel.hex", c2_w_int, 8)
write_hex("fc1_kernel.hex", f1_w_int, 8)
write_hex("fc2_kernel.hex", f2_w_int, 8)
write_hex("fc3_kernel.hex", f3_w_int, 8)


# Bias - dùng thống nhất 32-bit cho an toàn (không còn giới hạn 20/24-bit như bản cũ)
write_hex("conv1_bias.hex", c1_b_int, 32)
write_hex("conv2_bias.hex", c2_b_int, 32)
write_hex("fc1_bias.hex", f1_b_int, 32)
write_hex("fc2_bias.hex", f2_b_int, 32)
write_hex("fc3_bias.hex", f3_b_int, 32)


# Multiplier (16-bit unsigned) + Shift (8-bit unsigned) - MỚI, per-channel
write_hex("conv1_multiplier.hex", c1_mult, 16)
write_hex("conv1_shift.hex", c1_shift, 8)
write_hex("conv2_multiplier.hex", c2_mult, 16)
write_hex("conv2_shift.hex", c2_shift, 8)
write_hex("fc1_multiplier.hex", f1_mult, 16)
write_hex("fc1_shift.hex", f1_shift, 8)
write_hex("fc2_multiplier.hex", f2_mult, 16)
write_hex("fc2_shift.hex", f2_shift, 8)


print("\nĐã xuất trọng số + bias + multiplier/shift per-channel (kiểu TFLite) cho RTL.")




# ============================================================
# 23b. MÔ PHỎNG GOLDEN MODEL INT8 PER-CHANNEL (KIỂM TRA ACCURACY THẬT CỦA RTL MỚI)
# ============================================================


def requantize_per_channel(acc, mult_arr, shift_arr, num_channels):
    """
    acc: mảng int64 shape (channels, ...) hoặc (channels,).
    Áp dụng multiply-then-shift với multiplier/shift RIÊNG cho từng kênh
    (axis 0), có làm tròn (round-half-up trước khi shift phải), rồi ReLU
    và clip về [-128, 127] - đúng thứ tự mà RTL sẽ thực hiện.
    """
    out = np.zeros_like(acc)
    for c in range(num_channels):
        m = int(mult_arr[c])
        s = int(shift_arr[c])
        product = acc[c].astype(np.int64) * m
        if s > 0:
            product = product + (1 << (s - 1))
            val = product >> s
        else:
            val = product
        val = np.clip(val, -128, 127)
        val = np.maximum(val, 0)  # ReLU
        out[c] = val
    return out




def simulate_rtl_int8_v2(c1_w_int, c1_b_int, c1_mult, c1_shift,
                          c2_w_int, c2_b_int, c2_mult, c2_shift,
                          f1_w_int, f1_b_int, f1_mult, f1_shift,
                          f2_w_int, f2_b_int, f2_mult, f2_shift,
                          f3_w_int, f3_b_int,
                          S_in_conv1, test_dataset, num_samples=None):
    correct = 0
    total = 0


    for image_batch, label_batch in test_dataset.unbatch().batch(1):
        img = image_batch.numpy()[0, :, :, 0]
        label = int(label_batch.numpy()[0])


        x = np.clip(np.round(img * S_in_conv1), -128, 127).astype(np.int64)


        # CONV1
        acc1 = np.zeros((6, 26, 26), dtype=np.int64)
        for c in range(6):
            a = np.full((26, 26), c1_b_int[c], dtype=np.int64)
            for ky in range(3):
                for kx in range(3):
                    a += x[ky:ky + 26, kx:kx + 26] * c1_w_int[c, 0, ky, kx]
            acc1[c] = a
        c1 = requantize_per_channel(acc1, c1_mult, c1_shift, 6)


        p1 = c1.reshape(6, 13, 2, 13, 2).max(axis=(2, 4))


        # CONV2
        acc2 = np.zeros((16, 11, 11), dtype=np.int64)
        for c in range(16):
            a = np.full((11, 11), c2_b_int[c], dtype=np.int64)
            for kc in range(6):
                for ky in range(3):
                    for kx in range(3):
                        a += p1[kc, ky:ky + 11, kx:kx + 11] * c2_w_int[c, kc, ky, kx]
            acc2[c] = a
        c2 = requantize_per_channel(acc2, c2_mult, c2_shift, 16)


        # MaxPool2D valid trên 11x11 bỏ hàng/cột cuối, còn 5x5
        p2 = c2[:, :10, :10].reshape(16, 5, 2, 5, 2).max(axis=(2, 4))
        p2_flat = p2.reshape(400)


        # FC1
        acc_f1 = np.array([f1_b_int[c] + np.dot(p2_flat, f1_w_int[c]) for c in range(120)])
        fc1 = requantize_per_channel(acc_f1, f1_mult, f1_shift, 120)


        # FC2
        acc_f2 = np.array([f2_b_int[c] + np.dot(fc1, f2_w_int[c]) for c in range(84)])
        fc2 = requantize_per_channel(acc_f2, f2_mult, f2_shift, 84)


        # FC3 - raw, không requant
        fc3 = np.array([f3_b_int[c] + np.dot(fc2, f3_w_int[c]) for c in range(47)])


        pred = int(np.argmax(fc3))
        correct += int(pred == label)
        total += 1
        if num_samples is not None and total >= num_samples:
            break
        if total % 500 == 0:
            print(f"  Đã xử lý {total} ảnh, accuracy tạm thời: {correct / total * 100:.2f}%")


    return correct / total, total




print("\nĐang chạy golden model int8 PER-CHANNEL (multiply-shift kiểu TFLite)...")
print("(có thể mất vài phút vì mỗi ảnh chạy convolution bằng vòng lặp Python thuần)")


int8_perchannel_accuracy, n_evaluated_v2 = simulate_rtl_int8_v2(
    c1_w_int, c1_b_int, c1_mult, c1_shift,
    c2_w_int, c2_b_int, c2_mult, c2_shift,
    f1_w_int, f1_b_int, f1_mult, f1_shift,
    f2_w_int, f2_b_int, f2_mult, f2_shift,
    f3_w_int, f3_b_int,
    S_in_conv1, test_dataset, num_samples=2000,
)


print(f"\nSố ảnh đã đánh giá                       : {n_evaluated_v2}")
print(f"Accuracy int8 per-channel (mô phỏng RTL) : {int8_perchannel_accuracy * 100:.2f}%")
print(f"Accuracy float32 gốc                     : {test_accuracy * 100:.2f}%")
print(f"Accuracy TFLite int8 (đối chứng)         : {tflite_int8_accuracy * 100:.2f}%")
print(
    f"Chênh lệch do lượng tử hóa (per-channel)  : "
    f"{(test_accuracy - int8_perchannel_accuracy) * 100:.2f} điểm %"
)


with open(QUANT_REPORT_PATH, "a", encoding="utf-8") as file:
    file.write(f"\n# Accuracy int8 per-channel (RTL)  : {int8_perchannel_accuracy:.6f}")




# ============================================================
# 24. XUẤT TEST VECTOR CHO TESTBENCH RTL - NHÓM 1
# ============================================================


TEST_VECTORS_DIR = os.path.join(FPGA_EXPORT_DIR, "test_vectors")
os.makedirs(TEST_VECTORS_DIR, exist_ok=True)


NUM_TEST_VECTORS = 20


# QUAN TRỌNG: phải dùng đúng S_in_conv1 đã dùng để tính c1_bias ở mục 23
# (S_in_conv1 = 127.0, giả định input in [0,127], KHÔNG zero-center).
assert S_in_conv1 == 127.0, "S_in_conv1 đã đổi ở mục 23 nhưng chưa cập nhật lại đoạn này!"


vector_index = 0
for image_batch, label_batch in test_dataset.unbatch().batch(1).take(NUM_TEST_VECTORS):
    image_np = image_batch.numpy()[0]  # (28, 28, 1), float32 trong [0, 1]
    true_label = int(label_batch.numpy()[0])


    prediction = model.predict(image_batch, verbose=0)
    predicted_label_float = int(np.argmax(prediction[0]))


    image_int8 = np.clip(
        np.round(image_np[:, :, 0] * S_in_conv1), -128, 127
    ).astype(np.int8)


    vector_path = os.path.join(
        TEST_VECTORS_DIR,
        f"vector_{vector_index:03d}_label{true_label}_pred{predicted_label_float}.hex",
    )
    with open(vector_path, "w", encoding="utf-8") as file:
        for row in image_int8:
            for value in row:
                file.write(f"{int(value) & 0xFF:02X}\n")


    vector_index += 1


print(f"\nĐã xuất {vector_index} test vector cho testbench RTL tại:", TEST_VECTORS_DIR)
print(
    "Mỗi file .hex chứa 784 byte (28x28) theo thứ tự row-major, "
    "dùng $readmemh để nạp vào testbench Verilog."
)
print(
    "Test vector được lượng tử hóa bằng đúng S_in_conv1 dùng để tính bias "
    "(input in [0,127], không zero-center) - khớp với RTL per-channel."
)
print(
    "Lưu ý: 'pred' trong tên file là dự đoán của MODEL FLOAT32, dùng để "
    "đối chiếu logic; vì RTL giờ dùng lượng tử hóa per-channel khác hẳn "
    "bản cũ, một vài ảnh có margin nhỏ có thể ra kết quả khác 'pred' này "
    "dù RTL hoàn toàn đúng - hãy dùng int8_perchannel_accuracy ở trên làm "
    "thước đo chính, không dùng số PASS/FAIL từng ảnh làm tiêu chí duy nhất."
)




# ============================================================
# 25. BÁO CÁO TÀI NGUYÊN ƯỚC TÍNH (RESOURCE BUDGET) - NHÓM 1
# ============================================================


resource_report_lines = ["layer_name\tparams\tMACs_estimate"]
total_params = 0
total_macs = 0


conv_output_shapes = {
    "conv1": (26, 26, 6),
    "conv2": (11, 11, 16),
}


for layer in model.layers:
    layer_weights = layer.get_weights()
    if not layer_weights:
        continue
    kernel = layer_weights[0]
    num_params = int(np.prod(kernel.shape))
    total_params += num_params


    if layer.name in conv_output_shapes:
        out_h, out_w, out_c = conv_output_shapes[layer.name]
        kh, kw, in_c, out_c_k = kernel.shape
        macs = out_h * out_w * out_c * kh * kw * in_c
    else:
        macs = num_params  # Dense: 1 MAC / trọng số / ảnh


    total_macs += macs
    resource_report_lines.append(f"{layer.name}\t{num_params}\t{macs}")


resource_report_lines.append(f"TOTAL\t{total_params}\t{total_macs}")


RESOURCE_REPORT_PATH = os.path.join(
    FPGA_EXPORT_DIR,
    "resource_budget_estimate.tsv",
)
with open(RESOURCE_REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("\n".join(resource_report_lines))


print("\nĐã lưu ước tính tài nguyên (params/MACs) tại:", RESOURCE_REPORT_PATH)
print(f"Tổng số tham số : {total_params:,}")
print(f"Tổng số MAC/ảnh : {total_macs:,}")


if DIGITS_ONLY_FOR_FPGA:
    print(
        "\nLƯU Ý: DIGITS_ONLY_FOR_FPGA=True nhưng phần lọc dữ liệu chỉ-digit "
        "chưa được cài đặt ở bản này - cần lọc lại train/test dataset theo "
        "label < 10 và huấn luyện lại nếu muốn bản 10 lớp cho FPGA nhỏ."
    )


print("\nHoàn thành huấn luyện và đánh giá LeNet-5.")
print("Bộ dữ liệu: EMNIST Balanced, 47 lớp.")
print("Lượng tử hóa: PER-CHANNEL multiply-shift (kiểu TFLite), thay cho shift=8 cố định.")
print("Các kết quả được lưu trong thư mục:", OUTPUT_DIR)
print("Các sản phẩm phục vụ FPGA (Nhóm 1) lưu tại:", FPGA_EXPORT_DIR)



