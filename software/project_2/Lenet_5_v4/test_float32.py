"""Quick float32 accuracy test on all 1000 images"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

NEW_DIR = r"D:\THUC TAP\New folder"

model = tf.keras.models.load_model(os.path.join(NEW_DIR, "lenet5_emnist_3x3.keras"))

with open(os.path.join(NEW_DIR, "input_1000.mem"), 'r') as f:
    img_hex = [x.strip() for x in f.readlines() if x.strip()]
images_raw = np.array([int(x, 16) for x in img_hex], dtype=np.uint8)

with open(os.path.join(NEW_DIR, "label_1000.mem"), 'r') as f:
    label_hex = [x.strip() for x in f.readlines() if x.strip()]
labels = np.array([int(x, 16) for x in label_hex], dtype=np.int64)

# Test float32 model on all 1000 images (no transpose)
imgs = images_raw.reshape(1000, 28, 28, 1).astype(np.float32) / 255.0
preds = model.predict(imgs, batch_size=128, verbose=0)
pred_labels = np.argmax(preds, axis=1)

correct = np.sum(pred_labels == labels)
print(f"Float32 model on input_1000.mem (no transpose): {correct}/1000 = {correct/10:.1f}%")

# With transpose
imgs_t = np.transpose(imgs, (0, 2, 1, 3))
preds_t = model.predict(imgs_t, batch_size=128, verbose=0)
pred_labels_t = np.argmax(preds_t, axis=1)
correct_t = np.sum(pred_labels_t == labels)
print(f"Float32 model on input_1000.mem (with transpose): {correct_t}/1000 = {correct_t/10:.1f}%")

# Also test with EMNIST TFDS to confirm model accuracy
try:
    import tensorflow_datasets as tfds
    
    def preprocess(image, label):
        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    _, test_raw = tfds.load("emnist/balanced", split=["train", "test"],
                             as_supervised=True)
    
    test_ds = test_raw.map(preprocess).batch(128)
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\nFloat32 model on EMNIST TFDS test set: {acc*100:.2f}%")
except Exception as e:
    print(f"\nCould not test on TFDS: {e}")

print("\nDone!")
