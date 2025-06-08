import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import logging

# Setup logging
logging.basicConfig(filename=os.path.join('G:/PlantDisease', 'preprocess_log.txt'), level=logging.INFO)

# Paths
base_path = 'G:/PlantDisease'
data_dir = os.path.join(base_path, 'PlantifyDr')
tfrecord_dir = os.path.join(base_path, 'TFRecords')
batch_size = 8

# Create output directory
os.makedirs(tfrecord_dir, exist_ok=True)

# Detect classes
classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
num_classes = len(classes)
image_counts = {}
total_images = 0
for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_counts[class_name] = len(images)
    total_images += len(images)

# Log counts
logging.info(f"Number of classes: {num_classes}")
for class_name, count in image_counts.items():
    logging.info(f"Class {class_name}: {count} images")
logging.info(f"Total images: {total_images}")

# ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# Custom split logic
train_images = {}
val_images = {}
test_images = {}
for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    total = len(images)
    train_size = int(0.8 * total)
    test_size = int(0.1 * total)
    val_size = total - train_size - test_size
    train_images[class_name] = images[:train_size]
    val_images[class_name] = images[train_size:train_size + val_size]
    test_images[class_name] = images[train_size + val_size:train_size + val_size + test_size]
    logging.info(f"{class_name}: train={len(train_images[class_name])}, val={len(val_images[class_name])}, test={len(test_images[class_name])}")

# TFRecord creation
def create_tfrecord(image_path, label):
    try:
        img = tf.io.read_file(image_path)
        if image_path.lower().endswith(('.jpg', '.jpeg')):
            img = tf.image.decode_jpeg(img, channels=3)
        elif image_path.lower().endswith('.png'):
            img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy().tobytes()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None

def write_tfrecord(images, class_path, class_idx, writer):
    for img in images:
        img_path = os.path.join(class_path, img)
        example = create_tfrecord(img_path, class_idx)
        if example:
            writer.write(example.SerializeToString())

# Class indices
class_indices = {name: idx for idx, name in enumerate(classes)}

# Save training TFRecord
try:
    with tf.io.TFRecordWriter(os.path.join(tfrecord_dir, 'plantifydr_train.tfrecord')) as writer:
        for class_name, images in train_images.items():
            class_path = os.path.join(data_dir, class_name)
            class_idx = class_indices[class_name]
            write_tfrecord(images, class_path, class_idx, writer)
    logging.info("Training TFRecord saved")
except Exception as e:
    logging.error(f"Training TFRecord failed: {str(e)}")
    raise

# Save validation TFRecord
try:
    with tf.io.TFRecordWriter(os.path.join(tfrecord_dir, 'plantifydr_val.tfrecord')) as writer:
        for class_name, images in val_images.items():
            class_path = os.path.join(data_dir, class_name)
            class_idx = class_indices[class_name]
            write_tfrecord(images, class_path, class_idx, writer)
    logging.info("Validation TFRecord saved")
except Exception as e:
    logging.error(f"Validation TFRecord failed: {str(e)}")
    raise

# Save test TFRecord
try:
    with tf.io.TFRecordWriter(os.path.join(tfrecord_dir, 'plantifydr_test.tfrecord')) as writer:
        for class_name, images in test_images.items():
            class_path = os.path.join(data_dir, class_name)
            class_idx = class_indices[class_name]
            write_tfrecord(images, class_path, class_idx, writer)
    logging.info("Test TFRecord saved")
except Exception as e:
    logging.error(f"Test TFRecord failed: {str(e)}")
    raise

# Verify splits
train_count = sum(len(images) for images in train_images.values())
val_count = sum(len(images) for images in val_images.values())
test_count = sum(len(images) for images in test_images.values())
logging.info(f"Final splits: train={train_count}, val={val_count}, test={test_count}")
logging.info(f"Class indices: {class_indices}")

print(f"Preprocessing complete. Splits: train={train_count}, val={val_count}, test={test_count}")
logging.info("Preprocessing complete")