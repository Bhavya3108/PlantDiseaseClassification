import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Logging
logging.basicConfig(filename=os.path.join('G:/PlantDisease', 'evaluation_log.txt'), level=logging.INFO)

# GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("GPU memory growth enabled")
    except Exception as e:
        logging.error(f"GPU config failed: {str(e)}")

# Paths
base_path = 'G:/PlantDisease'
tfrecord_dir = os.path.join(base_path, 'TFRecords')
test_tfrecord = os.path.join(tfrecord_dir, 'plantifydr_test.tfrecord')
model_path = os.path.join(base_path, 'plantifydr_mobilenetv2.h5')

# Parameters
num_classes = 33
batch_size = 8
img_height, img_width = 224, 224

# Parse TFRecords
def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(example['image'], tf.float32)
    image = tf.reshape(image, [img_height, img_width, 3])
    label = tf.cast(example['label'], tf.int32)
    return image, tf.one_hot(label, num_classes)

# Load dataset
def load_dataset(tfrecord_path, batch_size, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(2)
    return dataset

try:
    test_dataset = load_dataset(test_tfrecord, batch_size, shuffle=False)
    logging.info("Test dataset loaded")
except Exception as e:
    logging.error(f"Test dataset loading failed: {str(e)}")
    raise

# Class counts
image_counts = {
    'AppleBlackRot': 3107, 'AppleHealthy': 4245, 'AppleRust': 2567, 'AppleScab': 3245,
    'BellPepperBacterialSpot': 3460, 'BellPepperHealthy': 4024, 'CherryHealthy': 3194,
    'CherryPowderyMildew': 3158, 'CornCommonRust': 3695, 'CornGrayLeafSpot': 2635,
    'CornHealthy': 3488, 'CornNorthernLeafBlight': 3562, 'GrapeBlackMeasles': 3786,
    'GrapeBlackRot': 3607, 'GrapeHealthy': 2610, 'GrapeIsariopsisLeafSpot': 3231,
    'PeachBacterialSpot': 4596, 'PeachHealthy': 2634, 'PotatoEarlyBlight': 3545,
    'PotatoHealthy': 2434, 'PotatoLateBlight': 3531, 'StrawberryHealthy': 2829,
    'StrawberryLeafScorch': 3329, 'TomatoBacterialSpot': 4366, 'TomatoEarlyBlight': 3493,
    'TomatoHealthy': 4066, 'TomatoLateBlight': 4335, 'TomatoLeafMold': 3397,
    'TomatoMosaicVirus': 2667, 'TomatoSeptoriaLeafSpot': 4104, 'TomatoSpiderMites': 3856,
    'TomatoTargetSpot': 3690, 'TomatoYellowLeafCurlVirus': 7889
}

# Load model
try:
    model = tf.keras.models.load_model(model_path)
    logging.info(f"Model loaded from {model_path}")
except Exception as e:
    logging.error(f"Model loading failed: {str(e)}")
    raise

# Evaluate
try:
    all_pred_labels = []
    all_true_labels = []
    predictions = []

    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        true_labels = np.argmax(labels.numpy(), axis=1)
        all_pred_labels.extend(pred_labels)
        all_true_labels.extend(true_labels)
        predictions.extend(preds)

    all_pred_labels = np.array(all_pred_labels)
    all_true_labels = np.array(all_true_labels)
    predictions = np.array(predictions)

    # Metrics
    accuracy = np.mean(all_pred_labels == all_true_labels)
    precision = precision_score(all_true_labels, all_pred_labels, average='weighted')
    recall = recall_score(all_true_labels, all_pred_labels, average='weighted')
    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # IoU
    iou_scores = []
    for c in range(num_classes):
        true_c = (all_true_labels == c)
        pred_c = (all_pred_labels == c)
        intersection = np.sum(true_c & pred_c)
        union = np.sum(true_c | pred_c)
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    mean_iou = np.mean(iou_scores)
    logging.info(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")

    # mAP
    ap_scores = []
    for c in range(num_classes):
        true_c = (all_true_labels == c)
        pred_c = (all_pred_labels == c)
        precision_c = precision_score(true_c, pred_c, zero_division=0)
        ap_scores.append(precision_c)
    mean_ap = np.mean(ap_scores)
    logging.info(f"mAP: {mean_ap:.4f}")
    print(f"mAP: {mean_ap:.4f}")

    # MSE
    mse = np.mean((predictions - tf.keras.utils.to_categorical(all_true_labels, num_classes)) ** 2)
    logging.info(f"MSE: {mse:.4f}")
    print(f"MSE: {mse:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(image_counts.keys()), yticklabels=sorted(image_counts.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(base_path, 'confusion_matrix.png'))
    plt.close()
    logging.info("Confusion matrix saved")

except Exception as e:
    logging.error(f"Evaluation failed: {str(e)}")
    raise