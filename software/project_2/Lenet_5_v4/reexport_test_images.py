"""
Re-export 1000 test images + labels from EMNIST TFDS
using the EXACT same preprocessing as training (correct_emnist_orientation).
Images are quantized to int8 using S_in_conv1 = 127.0 (range [0, 127]).
Output: input_1000.mem and label_1000.mem in hex format.
"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
tf.get_logger().setLevel('ERROR')

NEW_DIR = r"D:\THUC TAP\New folder"
SIM_DIR = r"D:\THUC TAP\Project_vivado\cnn_accelerator\cnn_accelerator.sim\sim_1\behav\xsim"

NUM_IMAGES = 1000

# ---- 1. Load model for verification ----
model = tf.keras.models.load_model(os.path.join(NEW_DIR, "lenet5_emnist_3x3.keras"))
print("Model loaded!")

# ---- 2. Load EMNIST TFDS test set ----
_, test_raw = tfds.load("emnist/balanced", split=["train", "test"], as_supervised=True)

def correct_emnist_orientation(image):
    return tf.transpose(image, perm=[1, 0, 2])

def preprocess_image(image, label):
    image = correct_emnist_orientation(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# ---- 3. Extract NUM_IMAGES test images ----
images_float = []
labels_list = []
images_uint8 = []  # For .mem export (pre-transposed uint8)

for image_raw, label in test_raw.take(NUM_IMAGES):
    # Apply same preprocessing as training
    img_corrected = correct_emnist_orientation(image_raw)
    img_uint8 = img_corrected.numpy()[:, :, 0]  # (28, 28) uint8, already transposed
    img_float = img_uint8.astype(np.float32) / 255.0
    
    images_float.append(img_float)
    images_uint8.append(img_uint8)
    labels_list.append(int(label.numpy()))

images_float = np.array(images_float)  # (1000, 28, 28)
labels_arr = np.array(labels_list)

print(f"Extracted {len(images_float)} images")
print(f"Label range: {labels_arr.min()} to {labels_arr.max()}")

# ---- 4. Verify float32 model on these images ----
imgs_4d = images_float.reshape(-1, 28, 28, 1)
preds = model.predict(imgs_4d, batch_size=128, verbose=0)
pred_labels = np.argmax(preds, axis=1)
correct_float = np.sum(pred_labels == labels_arr)
print(f"\nFloat32 model accuracy on new images: {correct_float}/{NUM_IMAGES} = {correct_float/NUM_IMAGES*100:.2f}%")

# ---- 5. Write input_1000.mem ----
# Each pixel is uint8 [0, 255], stored as 2-digit hex, one per line
# Total: 1000 * 784 = 784000 lines
input_lines = []
for i in range(NUM_IMAGES):
    for r in range(28):
        for c in range(28):
            val = int(images_uint8[i][r, c])
            input_lines.append(f"{val:02x}")

for output_path in [os.path.join(NEW_DIR, "input_1000.mem"),
                     os.path.join(SIM_DIR, "input_1000.mem")]:
    with open(output_path, 'w') as f:
        f.write("\n".join(input_lines) + "\n")
    print(f"Written input_1000.mem to: {output_path}")

# ---- 6. Write label_1000.mem ----
label_lines = [f"{int(l):02x}" for l in labels_arr]

for output_path in [os.path.join(NEW_DIR, "label_1000.mem"),
                     os.path.join(SIM_DIR, "label_1000.mem")]:
    with open(output_path, 'w') as f:
        f.write("\n".join(label_lines) + "\n")
    print(f"Written label_1000.mem to: {output_path}")

# ---- 7. Quick golden int8 model verification ----
print("\n=== Quick golden int8 verification (100 images) ===")

S_in_conv1 = 127.0

# Read freshly exported weights
def to_int8(val):
    val = val & 0xFF
    return val if val < 128 else val - 256

def to_int32(val):
    val = val & 0xFFFFFFFF
    return val if val < 0x80000000 else val - 0x100000000

def read_hex_8(fp):
    with open(fp) as f:
        return np.array([to_int8(int(x.strip(),16)) for x in f if x.strip()], dtype=np.int64)

def read_hex_32(fp):
    with open(fp) as f:
        return np.array([to_int32(int(x.strip(),16)) for x in f if x.strip()], dtype=np.int64)

def read_hex_16u(fp):
    with open(fp) as f:
        return np.array([int(x.strip(),16) for x in f if x.strip()], dtype=np.int64)

def read_hex_8u(fp):
    with open(fp) as f:
        return np.array([int(x.strip(),16)&0xFF for x in f if x.strip()], dtype=np.int64)

MEM = os.path.join(NEW_DIR, "memory_files")
c1_w = read_hex_8(os.path.join(MEM,"conv1_kernel.mem")).reshape(6,1,3,3)
c1_b = read_hex_32(os.path.join(MEM,"conv1_bias.mem"))
c1_m = read_hex_16u(os.path.join(MEM,"conv1_multiplier.mem"))
c1_s = read_hex_8u(os.path.join(MEM,"conv1_shift.mem"))

c2_w = read_hex_8(os.path.join(MEM,"conv2_kernel.mem")).reshape(16,6,3,3)
c2_b = read_hex_32(os.path.join(MEM,"conv2_bias.mem"))
c2_m = read_hex_16u(os.path.join(MEM,"conv2_multiplier.mem"))
c2_s = read_hex_8u(os.path.join(MEM,"conv2_shift.mem"))

f1_w = read_hex_8(os.path.join(MEM,"fc1_kernel.mem")).reshape(120,400)
f1_b = read_hex_32(os.path.join(MEM,"fc1_bias.mem"))
f1_m = read_hex_16u(os.path.join(MEM,"fc1_multiplier.mem"))
f1_s = read_hex_8u(os.path.join(MEM,"fc1_shift.mem"))

f2_w = read_hex_8(os.path.join(MEM,"fc2_kernel.mem")).reshape(84,120)
f2_b = read_hex_32(os.path.join(MEM,"fc2_bias.mem"))
f2_m = read_hex_16u(os.path.join(MEM,"fc2_multiplier.mem"))
f2_s = read_hex_8u(os.path.join(MEM,"fc2_shift.mem"))

f3_w = read_hex_8(os.path.join(MEM,"fc3_kernel.mem")).reshape(47,84)
f3_b = read_hex_32(os.path.join(MEM,"fc3_bias.mem"))

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

correct_int8 = 0
for i in range(100):
    img_f = images_uint8[i].astype(np.float32) / 255.0
    x = np.clip(np.round(img_f * S_in_conv1), -128, 127).astype(np.int64)
    
    o1 = np.zeros((6,26,26),dtype=np.int64)
    for oc in range(6):
        for r in range(26):
            for cc in range(26):
                a = int(c1_b[oc])
                for ky in range(3):
                    for kx in range(3):
                        a += int(x[r+ky,cc+kx])*int(c1_w[oc,0,ky,kx])
                o1[oc,r,cc] = rq(a,c1_m[oc],c1_s[oc])
    
    p1 = o1.reshape(6,13,2,13,2).max(axis=(2,4))
    
    o2 = np.zeros((16,11,11),dtype=np.int64)
    for oc in range(16):
        for r in range(11):
            for cc in range(11):
                a = int(c2_b[oc])
                for ic in range(6):
                    for ky in range(3):
                        for kx in range(3):
                            a += int(p1[ic,r+ky,cc+kx])*int(c2_w[oc,ic,ky,kx])
                o2[oc,r,cc] = rq(a,c2_m[oc],c2_s[oc])
    
    p2 = o2[:,:10,:10].reshape(16,5,2,5,2).max(axis=(2,4))
    pf = p2.reshape(400)
    
    o3 = np.zeros(120,dtype=np.int64)
    for n in range(120):
        a = int(f1_b[n]) + int(np.dot(pf,f1_w[n]))
        o3[n] = rq(a,f1_m[n],f1_s[n])
    
    o4 = np.zeros(84,dtype=np.int64)
    for n in range(84):
        a = int(f2_b[n]) + int(np.dot(o3,f2_w[n]))
        o4[n] = rq(a,f2_m[n],f2_s[n])
    
    o5 = np.zeros(47,dtype=np.int64)
    for n in range(47):
        a = int(f3_b[n]) + int(np.dot(o4,f3_w[n]))
        o5[n] = a
    
    pred = int(np.argmax(o5))
    if pred == labels_arr[i]:
        correct_int8 += 1

print(f"Golden int8 model accuracy: {correct_int8}/100 = {correct_int8}%")

# Also verify the RTL input format (>> 1)
correct_shift = 0
for i in range(100):
    x = (images_uint8[i].astype(np.int64) >> 1)  # This is what RTL testbench does
    
    o1 = np.zeros((6,26,26),dtype=np.int64)
    for oc in range(6):
        for r in range(26):
            for cc in range(26):
                a = int(c1_b[oc])
                for ky in range(3):
                    for kx in range(3):
                        a += int(x[r+ky,cc+kx])*int(c1_w[oc,0,ky,kx])
                o1[oc,r,cc] = rq(a,c1_m[oc],c1_s[oc])
    
    p1 = o1.reshape(6,13,2,13,2).max(axis=(2,4))
    
    o2 = np.zeros((16,11,11),dtype=np.int64)
    for oc in range(16):
        for r in range(11):
            for cc in range(11):
                a = int(c2_b[oc])
                for ic in range(6):
                    for ky in range(3):
                        for kx in range(3):
                            a += int(p1[ic,r+ky,cc+kx])*int(c2_w[oc,ic,ky,kx])
                o2[oc,r,cc] = rq(a,c2_m[oc],c2_s[oc])
    
    p2 = o2[:,:10,:10].reshape(16,5,2,5,2).max(axis=(2,4))
    pf = p2.reshape(400)
    
    o3 = np.zeros(120,dtype=np.int64)
    for n in range(120):
        a = int(f1_b[n]) + int(np.dot(pf,f1_w[n]))
        o3[n] = rq(a,f1_m[n],f1_s[n])
    
    o4 = np.zeros(84,dtype=np.int64)
    for n in range(84):
        a = int(f2_b[n]) + int(np.dot(o3,f2_w[n]))
        o4[n] = rq(a,f2_m[n],f2_s[n])
    
    o5 = np.zeros(47,dtype=np.int64)
    for n in range(47):
        a = int(f3_b[n]) + int(np.dot(o4,f3_w[n]))
        o5[n] = a
    
    pred = int(np.argmax(o5))
    if pred == labels_arr[i]:
        correct_shift += 1

print(f"Golden int8 model (>> 1 input): {correct_shift}/100 = {correct_shift}%")

print("\n=== SUMMARY ===")
print(f"Float32 model on correct EMNIST test: {correct_float}/{NUM_IMAGES} = {correct_float/NUM_IMAGES*100:.1f}%")
print(f"Int8 golden (input*127/255):          {correct_int8}/100 = {correct_int8}%")
print(f"Int8 golden (input >> 1):             {correct_shift}/100 = {correct_shift}%")
print("\nAll files have been re-exported!")
