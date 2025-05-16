# 🚦 Traffic Sign Recognition Project

This project implements a **traffic sign recognition system** using deep learning models. It processes the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset to classify traffic signs into **43 distinct categories**.

Multiple convolutional neural network (CNN) architectures are used, including:
- **AlexNet**
- **VGGNet**
- **Custom CNN**

---

## 🧰 Technologies Used

- **Python**: Core programming language for data processing and model implementation.
- **NumPy**: Numerical computations and array manipulations.
- **Pandas**: Handling the sign names CSV file.
- **Matplotlib**: Data visualization (class distribution, model performance).
- **Seaborn**: Creating confusion matrix heatmaps.
- **OpenCV (cv2)**: Image transformations (e.g., translation).
- **Scikit-Image**: Advanced image transformations (rotation, projective transforms).
- **TensorFlow/Keras**: Building and training deep learning models.
- **Scikit-Learn**: Data shuffling, confusion matrices, classification reports.

---

## 📁 Dataset

The project uses the **GTSRB dataset**, split into:

- **Training Set**: 34,799 images  
- **Validation Set**: 4,410 images  
- **Test Set**: 12,630 images  

Each image is a `32x32x3` RGB image. The dataset contains **43 unique traffic sign classes**.

---

## 🔧 Data Preprocessing

- **Loading**: Datasets are loaded from pickled files (`train.p`, `valid.p`, `test.p`).
- **Visualization**: Class distributions plotted and random samples visualized.
- **Augmentation**: Images augmented via rotation, translation, and projective transforms (target: 5,000 samples/class).
- **Grayscale Conversion**: RGB ➝ Grayscale (for AlexNet & Custom CNN).
- **Normalization**: Grayscale images normalized to range `[-1, 1]`.
- **One-Hot Encoding**: Labels converted for multi-class classification.

---

## 🧠 Models

### 🔷 AlexNet
- **Input**: `32x32x1` (grayscale)
- **Architecture**: 5 Conv layers, 3 MaxPooling layers, 3 Dense layers with Dropout
- **Optimizer**: SGD
- **Epochs**: 10

### 🔷 VGGNet
- **Input**: `32x32x3` (RGB)
- **Architecture**: 13 Conv layers, 5 MaxPooling layers, 3 Dense layers
- **Optimizer**: SGD
- **Training**: Early stopping & model checkpointing

### 🔷 Custom CNN
- **Input**: `32x32x1` (grayscale)
- **Architecture**: 6 Conv layers, 3 MaxPooling layers, 2 Dense layers with Dropout
- **Optimizer**: Adam (lr=0.001)
- **Training**: Early stopping & model checkpointing

---

## 📊 Training and Evaluation

- **Training**: Monitored using validation loss
- **Evaluation**:
  - Accuracy & loss plots
  - Confusion matrices
  - Classification reports
- **Test Accuracy**: Printed in the notebook for the custom CNN model
- **Visualization**: Real vs predicted labels displayed
