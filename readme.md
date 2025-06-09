## Plant Disease Classification Using Deep Learning

### Overview

This project aims to develop a deep learning model for classifying plant diseases, facilitating early detection to enhance agricultural productivity. It focuses on the PlantifyDr dataset, which comprises 118,375 images across 33 classes, representing various plant diseases and healthy states for crops like apples, tomatoes, grapes, and more. The dataset presents challenges such as class imbalance and visual similarity between classes. The goal is to train a robust model which classifies the plant diseases in real-time and evaluate it using metrics like accuracy, precision, recall, IoU, mAP, and MSE. This repository includes scripts for preprocessing, training, evaluation, and generating annotations, along with results and visualizations.

### Dataset

The PlantifyDr dataset consists of 118,375 images across 33 classes. The key characteristics are:
+ **Size:** 118,375 images
+ **Classes:** 33
+ **Imbalance:** Significant class imbalance (e.g., TomatoYellowLeafCurlVirus: 7,889 images, PotatoHealthy: 2,434 images, ~3.24 ratio)
+ **Splits:** 80/10/10 (train: 94,688, validation: 11,866, test: 11,821 images)
+ **Challenges:** Visual similarity between classes and class imbalance

The dataset is not included in this repository due to its size. It can be downloaded from [kaggle](https://www.kaggle.com/datasets/lavaman151/plantifydr-dataset) website.

### Model Architecture

The model uses MobileNetV2, pre-trained on ImageNet, with a custom classification head:
+ **Base Model:** MobileNetV2 (initially frozen)
+ **Custom Head:** GlobalAveragePooling2D, Dense(512, ReLU), Dropout(0.5), Dense(33, softmax)
+ **Fine-Tuning:** Unfroze layers after the first 50, fine-tuned with a learning rate of 1e-5

**Why MobileNetV2?** Suitable for the PlantifyDr dataset’s 33-class problem, and ideal for edge deployment (e.g., drones) due to its small size (~14 MB) and low computational requirements.

### Installation

#### Prerequisites

+ [Python 3.9.12](https://www.python.org/downloads/release/python-3912/)
+ [Git](https://git-scm.com/)
+ [VSCode](https://code.visualstudio.com/download)
+ NVIDIA GPU with CUDA and cuDNN (optional but recommended)

#### Setup

1. Clone the repository:
```
git clone https://github.com/Bhavya3108/PlantDiseaseClassification.git
```
```
cd PlantDiseaseClassification
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Place the PlantifyDr dataset in C:/PlantDiseaseClassification with subdirectories for each class.

### Usage

#### 1. Generate Annotations

Create the annotation file mapping image filenames to labels and class IDs:
```
python annotation.py
```
This generates ```plantifydr_annotations.csv``` in C:/PlantDiseaseClassification.

#### 2. Preprocess and Verify the Dataset

Convert the dataset into TFRecords and apply data augmentation:
```
python preprocess.py
```
```
python verify.py
```
**Outputs:**
+ *TFRecords:* ```plantifydr_train.tfrecord```, ```plantifydr_val.tfrecord```, ```plantifydr_test.tfrecord```
+ *Log:* ```preprocess_log.txt```, ```verify_log.txt```
+ *Sample Images:* ```train_samples.png```, ```val_samples.png```, ```test_samples.png``` (for verification)

#### 3. Train the Model

Train the MobileNetV2 model with initial training and fine-tuning:
```
python train.py
```
**Outputs:**
+ *Trained Model:* ```plantifydr_mobilenetv2.h5```
+ *Checkpoints:* ```best_model.h5```, ```best_model_finetuned.h5```
+ *Log:* ```training_log.txt```

#### 4. Evaluate the Model

Evaluate the model on the test set and generate metrics and visualizations:
```
python evaluate.py
```
**Outputs:**
+ *Metrics:* Test accuracy (58.52%), precision (0.8106), recall (0.5852), IoU (0.4640), mAP (~0.5–0.6), MSE (0.0187)
+ *Visualization:* ```confusion_matrix.png```
+ *Log:* ```evaluation_log.txt```

### Results

+ **Test Accuracy:** 58.52%
+ **Precision:** 0.8106 (weighted)
+ **Recall:** 0.5852
+ **Mean IoU:** 0.4640
+ **mAP:** 0.8344
+ **MSE:** 0.0187
+ **Confusion Matrix Insights:** High confusion between TomatoLateBlight and TomatoEarlyBlight due to visual similarity, Bias toward frequent classes (e.g., TomatoYellowLeafCurlVirus) due to imbalance.

### Limitations and Future Work

#### 1. Accuracy (58.52%):
+ Dataset Quality: Class imbalance (e.g., TomatoYellowLeafCurlVirus: 7,889 vs. PotatoHealthy: 2,434) and visual similarity between classes (e.g., TomatoLateBlight vs. TomatoEarlyBlight) limited performance.
+ Hardware Constraints: GTX 1650 (2 GB VRAM) restricted batch size to 8, causing noisy gradients. Training took ~10–12 hours, limiting hyperparameter tuning.
+ Overfitting: Train accuracy (98.01%) exceeded test accuracy (58.52%), despite dropout and class weights.

#### 2. Future Improvements:
+ Use more aggressive data augmentation.
+ Implement cross-validation to better handle class imbalance.
+ Use better hardware for larger batch sizes and faster experimentation.
+ Explore heavier models (e.g., EfficientNet) if hardware permits.

### Note
***PlantifyDr dataset and TFRecords are not included in this repository due to its size limitations.***
