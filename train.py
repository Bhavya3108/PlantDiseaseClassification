import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import logging

# Logging
logging.basicConfig(filename=os.path.join('G:/PlantDisease', 'training_log.txt'), level=logging.INFO)

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
train_tfrecord = os.path.join(tfrecord_dir, 'plantifydr_train.tfrecord')
val_tfrecord = os.path.join(tfrecord_dir, 'plantifydr_val.tfrecord')

# Parameters
num_classes = 33
batch_size = 8
img_height, img_width = 224, 224
epochs = 20
learning_rate = 1e-4

# Parse TFRecords
def parse_tfrecord(example_proto, training=False):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(example['image'], tf.float32)
    image = tf.reshape(image, [img_height, img_width, 3])
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    label = tf.cast(example['label'], tf.int32)
    return image, tf.one_hot(label, num_classes)

# Load datasets
def load_dataset(tfrecord_path, batch_size, shuffle=True, training=False):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(lambda x: parse_tfrecord(x, training=training), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(2)
    return dataset

try:
    train_dataset = load_dataset(train_tfrecord, batch_size, training=True)
    val_dataset = load_dataset(val_tfrecord, batch_size, shuffle=False)
    logging.info("Datasets loaded")
except Exception as e:
    logging.error(f"Dataset loading failed: {str(e)}")
    raise

# Inspect dataset
try:
    for images, labels in train_dataset.take(1):
        logging.info(f"Sample image shape: {images.shape}, label shape: {labels.shape}")
        logging.info(f"Image min: {tf.reduce_min(images)}, max: {tf.reduce_max(images)}")
except Exception as e:
    logging.error(f"Dataset inspection failed: {str(e)}")
    raise

# Class weights
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
class_indices = {name: idx for idx, name in enumerate(sorted(image_counts.keys()))}
class_weights = {class_indices[name]: min(max(image_counts.values()) / count, 10.0) for name, count in image_counts.items()}

# Model
try:
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    logging.info("Model built")
except Exception as e:
    logging.error(f"Model building failed: {str(e)}")
    raise

# Compile
model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
try:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(base_path, 'best_model.h5'), save_best_only=True)
        ]
    )
    logging.info(f"Initial training history: {history.history}")
except Exception as e:
    logging.error(f"Training failed: {str(e)}")
    raise

# Fine-tune
try:
    base_model.trainable = True
    for layer in base_model.layers[:50]:
        layer.trainable = False
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(base_path, 'best_model_finetuned.h5'), save_best_only=True)
        ]
    )
    logging.info(f"Fine-tuning history: {history.history}")
except Exception as e:
    logging.error(f"Fine-tuning failed: {str(e)}")
    raise

# Save
model.save(os.path.join(base_path, 'plantifydr_mobilenetv2.h5'))
logging.info(f"Model saved to {os.path.join(base_path, 'plantifydr_mobilenetv2.h5')}")