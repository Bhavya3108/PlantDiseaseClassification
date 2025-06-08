import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

# Logging
logging.basicConfig(filename=os.path.join('G:/PlantDisease', 'verify_log.txt'), level=logging.INFO)

# Paths
tfrecord_dir = 'G:/PlantDisease/TFRecords'
tfrecords = {
    'train': os.path.join(tfrecord_dir, 'plantifydr_train.tfrecord'),
    'val': os.path.join(tfrecord_dir, 'plantifydr_val.tfrecord'),
    'test': os.path.join(tfrecord_dir, 'plantifydr_test.tfrecord')
}
preprocess_log = 'G:/PlantDisease/preprocess_log.txt'

# Classes
classes = sorted([
    'AppleBlackRot', 'AppleHealthy', 'AppleRust', 'AppleScab',
    'BellPepperBacterialSpot', 'BellPepperHealthy', 'CherryHealthy',
    'CherryPowderyMildew', 'CornCommonRust', 'CornGrayLeafSpot',
    'CornHealthy', 'CornNorthernLeafBlight', 'GrapeBlackMeasles',
    'GrapeBlackRot', 'GrapeHealthy', 'GrapeIsariopsisLeafSpot',
    'PeachBacterialSpot', 'PeachHealthy', 'PotatoEarlyBlight',
    'PotatoHealthy', 'PotatoLateBlight', 'StrawberryHealthy',
    'StrawberryLeafScorch', 'TomatoBacterialSpot', 'TomatoEarlyBlight',
    'TomatoHealthy', 'TomatoLateBlight', 'TomatoLeafMold',
    'TomatoMosaicVirus', 'TomatoSeptoriaLeafSpot', 'TomatoSpiderMites',
    'TomatoTargetSpot', 'TomatoYellowLeafCurlVirus'
])

# Load class_indices
try:
    with open(preprocess_log, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Class indices' in line:
                class_indices = eval(line.split('Class indices: ')[1])
                logging.info(f"Loaded class_indices: {class_indices}")
                break
        else:
            raise ValueError("Class indices not found in preprocess_log.txt")
except Exception as e:
    logging.error(f"Failed to load class_indices: {str(e)}")
    raise

# Verify indices
if sorted(class_indices.keys()) != classes:
    logging.error(f"Class indices mismatch: expected {classes}, got {sorted(class_indices.keys())}")
    raise ValueError("Class indices mismatch")

# Parse TFRecord
def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(example['image'], tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    label = tf.cast(example['label'], tf.int32)
    return image, label

# Count samples
for split, tfrecord_path in tfrecords.items():
    try:
        dataset = tf.data.TFRecordDataset(tfrecord_path).map(parse_tfrecord)
        count = sum(1 for _ in dataset)
        labels = [label.numpy() for _, label in dataset]
        label_counts = {classes[i]: labels.count(i) for i in range(len(classes))}
        logging.info(f"{split.capitalize()} samples: {count}")
        for cls, cnt in label_counts.items():
            logging.info(f"{split.capitalize()} - {cls}: {cnt} samples")
        for label in labels:
            if label not in class_indices.values():
                logging.error(f"Invalid label {label} in {split} TFRecord")
                raise ValueError(f"Invalid label {label}")
    except Exception as e:
        logging.error(f"{split.capitalize()} counting failed: {str(e)}")
        raise

# Visualize
for split, tfrecord_path in tfrecords.items():
    try:
        dataset = tf.data.TFRecordDataset(tfrecord_path).map(parse_tfrecord).take(5)
        plt.figure(figsize=(15, 3))
        for i, (image, label) in enumerate(dataset):
            plt.subplot(1, 5, i+1)
            plt.imshow(image.numpy())
            plt.title(f"{classes[label.numpy()]}")
            plt.axis('off')
            logging.info(f"{split.capitalize()} sample {i}: min={tf.reduce_min(image):.4f}, max={tf.reduce_max(image):.4f}")
        plt.savefig(os.path.join('G:/PlantDisease', f'{split}_samples.png'))
        plt.close()
    except Exception as e:
        logging.error(f"{split.capitalize()} visualization failed: {str(e)}")
        raise

print("Verification complete. Check verify_log.txt and sample images.")
logging.info("Verification complete")