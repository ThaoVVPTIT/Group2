"""
Simulate exact RTL logic of all layers using the .mem files.
"""
import numpy as np
import os

NEW_DIR = r"D:\THUC TAP\New folder"
MEM_DIR = os.path.join(NEW_DIR, "memory_files")

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

c1_w = read_hex_8(os.path.join(MEM_DIR,"conv1_kernel.mem")).reshape(6,1,3,3)
c1_b = read_hex_32(os.path.join(MEM_DIR,"conv1_bias.mem"))
c1_m = read_hex_16u(os.path.join(MEM_DIR,"conv1_multiplier.mem"))
c1_s = read_hex_8u(os.path.join(MEM_DIR,"conv1_shift.mem"))

c2_w = read_hex_8(os.path.join(MEM_DIR,"conv2_kernel.mem")).reshape(16,6,3,3)
c2_b = read_hex_32(os.path.join(MEM_DIR,"conv2_bias.mem"))
c2_m = read_hex_16u(os.path.join(MEM_DIR,"conv2_multiplier.mem"))
c2_s = read_hex_8u(os.path.join(MEM_DIR,"conv2_shift.mem"))

f1_w = read_hex_8(os.path.join(MEM_DIR,"fc1_kernel.mem")).reshape(120,400)
f1_b = read_hex_32(os.path.join(MEM_DIR,"fc1_bias.mem"))
f1_m = read_hex_16u(os.path.join(MEM_DIR,"fc1_multiplier.mem"))
f1_s = read_hex_8u(os.path.join(MEM_DIR,"fc1_shift.mem"))

f2_w = read_hex_8(os.path.join(MEM_DIR,"fc2_kernel.mem")).reshape(84,120)
f2_b = read_hex_32(os.path.join(MEM_DIR,"fc2_bias.mem"))
f2_m = read_hex_16u(os.path.join(MEM_DIR,"fc2_multiplier.mem"))
f2_s = read_hex_8u(os.path.join(MEM_DIR,"fc2_shift.mem"))

f3_w = read_hex_8(os.path.join(MEM_DIR,"fc3_kernel.mem")).reshape(47,84)
f3_b = read_hex_32(os.path.join(MEM_DIR,"fc3_bias.mem"))

# Load test inputs
with open(os.path.join(NEW_DIR, "input_1000.mem"), 'r') as f:
    img_hex = [x.strip() for x in f.readlines() if x.strip()]
images_raw = np.array([int(x, 16) for x in img_hex], dtype=np.uint8)

with open(os.path.join(NEW_DIR, "label_1000.mem"), 'r') as f:
    label_hex = [x.strip() for x in f.readlines() if x.strip()]
labels = np.array([int(x, 16) for x in label_hex], dtype=np.int64)

# Print first predictions and labels
preds = []
for i in range(50):
    # s_tdata = images[...] >> 1
    x = (images_raw[i*784:(i+1)*784].reshape(28,28).astype(np.int64) >> 1)
    
    # Conv1: conv_calc + relu
    o1 = np.zeros((6,26,26), dtype=np.int64)
    for oc in range(6):
        for r in range(26):
            for cc in range(26):
                a = int(c1_b[oc])
                for ky in range(3):
                    for kx in range(3):
                        a += int(x[r+ky, cc+kx]) * int(c1_w[oc, 0, ky, kx])
                # quant logic in conv_calc
                prod = a * int(c1_m[oc])
                s = int(c1_s[oc])
                if s > 0:
                    prod = prod + (1 << (s - 1))
                    q = prod >> s
                else:
                    q = prod
                if q > 127: q = 127
                elif q < -128: q = -128
                if q < 0: q = 0 # ReLU
                o1[oc, r, cc] = q
    
    # Pool1 (maxpool_relu 26x26 -> 13x13)
    p1 = o1.reshape(6, 13, 2, 13, 2).max(axis=(2, 4))
    
    # Conv2 (conv_calc + relu)
    o2 = np.zeros((16,11,11), dtype=np.int64)
    for oc in range(16):
        for r in range(11):
            for cc in range(11):
                a = int(c2_b[oc])
                for ic in range(6):
                    for ky in range(3):
                        for kx in range(3):
                            a += int(p1[ic, r+ky, cc+kx]) * int(c2_w[oc, ic, ky, kx])
                prod = a * int(c2_m[oc])
                s = int(c2_s[oc])
                if s > 0:
                    prod = prod + (1 << (s - 1))
                    q = prod >> s
                else:
                    q = prod
                if q > 127: q = 127
                elif q < -128: q = -128
                if q < 0: q = 0 # ReLU
                o2[oc, r, cc] = q
    
    # Pool2 (maxpool_relu 11x11 -> 5x5, drop 11th row/col)
    p2 = o2[:, :10, :10].reshape(16, 5, 2, 5, 2).max(axis=(2, 4))
    
    # Flatten: ch*25 + pos_cnt (c*25 + y*5 + x)
    pf = p2.reshape(400)
    
    # FC1 (fc1_seq)
    o3 = np.zeros(120, dtype=np.int64)
    for n in range(120):
        a = int(f1_b[n]) + int(np.dot(pf, f1_w[n]))
        prod = a * int(f1_m[n])
        s = int(f1_s[n])
        if s > 0:
            prod = prod + (1 << (s - 1))
            acc = prod >> s
        else:
            acc = prod
        if acc < 0: acc = 0
        if acc > 127: acc = 127
        elif acc < -128: acc = -128
        o3[n] = acc
        
    # FC2 (fc2_seq)
    o4 = np.zeros(84, dtype=np.int64)
    for n in range(84):
        a = int(f2_b[n]) + int(np.dot(o3, f2_w[n]))
        prod = a * int(f2_m[n])
        s = int(f2_s[n])
        if s > 0:
            prod = prod + (1 << (s - 1))
            acc = prod >> s
        else:
            acc = prod
        if acc < 0: acc = 0
        if acc > 127: acc = 127
        elif acc < -128: acc = -128
        o4[n] = acc
        
    # FC3 (fc3_seq)
    o5 = np.zeros(47, dtype=np.int64)
    for n in range(47):
        a = int(f3_b[n]) + int(np.dot(o4, f3_w[n]))
        o5[n] = a
        
    pred = int(np.argmax(o5))
    preds.append(pred)

preds = np.array(preds)
correct = np.sum(preds == labels[:50])
print(f"Total correct: {correct}/50 ({correct/50.0*100:.1f}%)")
print("First 30 predictions: ", preds[:30])
print("First 30 labels:      ", labels[:30])

# Check frequency of predictions
unique, counts = np.unique(preds, return_counts=True)
print("\nPrediction distribution:")
for u, c in zip(unique, counts):
    print(f"  Class {u:2d}: {c:3d} times")
