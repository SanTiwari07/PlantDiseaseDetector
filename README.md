# 🌿 Plant Disease Detector

A deep learning-based system for detecting plant diseases from leaf images using **EfficientNetB3** and transfer learning. This project features a robust image classification model paired with an intuitive, interactive web interface built in Streamlit.

---

## 📖 Overview

The Plant Disease Detector helps agriculture professionals and enthusiasts accurately identify crop diseases from photographs of plant leaves. The model was trained on a comprehensive dataset encompassing 59 distinct plant and disease classifications, covering vital crops such as Tomato, Wheat, Sugarcane, Soybean, and more. 

- **Total Classes:** 59
- **Total Images:** 40,238
- **Distribution:** Balanced dataset with roughly ~700 images per class.

<details>
<summary><strong>View All 59 Detected Classes</strong></summary>

*(A full list is also available in `classes.txt`)*

- Bean_Angular_Leaf_Spot
- Bean_Healthy
- Bean_Rust
- Cotton_Bacterial_Blight
- Cotton_Fusarium_Wilt
- Cotton_Healthy
- Cotton_Leaf_Curl_Virus
- Grape_Bacterial_Rot
- Grape_Black_Rot
- Grape_Downy_Mildew
- Grape_Esca_Black_Measles
- Grape_Healthy
- Grape_Leaf_Blight
- Grape_Powdery_Mildew
- Groundnut_Alternaria_Leaf_Spot
- Groundnut_Early_Leaf_Spot
- Groundnut_Healthy
- Groundnut_Rosette
- Groundnut_Rust
- Maize_Cercospora_Leaf_Spot_Gray
- Maize_Common_Rust
- Maize_Healthy
- Maize_Northern_Leaf_Blight
- Millet_Blast
- Millet_Healthy
- Millet_Rust
- Potato_Early_Blight
- Potato_Healthy
- Potato_Late_Blight
- Sorghum_Anthracnose
- Sorghum_Head_Smut
- Sorghum_Healthy
- Sorghum_Kernel_Smut
- Sorghum_Loose_Smut
- Sorghum_Rust
- Soybean_Healthy
- Soybean_Powdery_Mildew
- Soybean_Sudden_Death_Syndrome
- Soybean_Yellow_Mosaic
- Sugarcane_Bacterial_Blight
- Sugarcane_Healthy
- Sugarcane_Mosaic
- Sugarcane_Red_Rot
- Sugarcane_Rust
- Sugarcane_Yellow
- Tomato_Bacterial_Spot
- Tomato_Early_Blight
- Tomato_Healthy
- Tomato_Late_Blight
- Tomato_Leaf_Curl_Virus
- Tomato_Leaf_Mold
- Tomato_Mosaic_Virus
- Tomato_Septoria_Leaf_Spot
- Tomato_Spider_Mites
- Tomato_Target_Spot
- Wheat_Crown_and_Root_Rot
- Wheat_Healthy
- Wheat_Leaf_Rust
- Wheat_Loose_Smut
</details>

---

## ✨ Key Features
- **Broad Multi-Crop Coverage:** Identifies agricultural diseases across an array of diverse plant varieties.
- **Robust Architecture:** Powered by `EfficientNetB3`, utilizing customized top dense layers built for high accuracy classification.
- **Dataset Augmentation:** Maintains balanced class distribution for resilient, bias-free predictions.
- **Interactive UI:** A real-time web application enables effortless image upload, automatic tensor preprocessing, and instantaneous predictions.

---

## 🚀 Installation & Usage

### 1. Requirements
Ensure you have Python 3.x installed. The required packages for running the local interface are:
- `streamlit`
- `tensorflow` (>= 2.0)
- `Pillow`
- `numpy`

### 2. Setup
Clone this repository locally. Verify that the pre-trained weights file (`plant_disease_model.keras`) and the class mapping configuration (`class_names.json`) are present in your project directory.

### 3. Running the App
To launch the Streamlit web interface locally, open your terminal in the project directory and run:
```bash
streamlit run app.py
```
Upload any supported leaf image (`.jpg`, `.png`) to the dashboard to receive the top 3 class predictions and an overall disease confidence score.

---

## 🧠 Model Architecture & Training

The architecture uses **EfficientNetB3** (pretrained on ImageNet) to extract deep visual features from leaf samples.

### Architecture Flow
1. **Base Model:** `EfficientNetB3`
2. **Pooling:** `GlobalAveragePooling2D()`
3. **Normalization:** `BatchNormalization()`
4. **Dense Layer 1:** 512 units (ReLU) + `Dropout(0.4)`
5. **Dense Layer 2:** 256 units (ReLU) + `Dropout(0.3)`
6. **Output:** 59 units (Softmax)
- **Total Parameters:** ~11.7 Million

### Training Strategy
Trained on Google Colab leveraging GPU hardware using TensorFlow 2.19. Images are preprocessed dynamically strictly via `efficientnet.preprocess_input` matching a default size of **224 × 224** and a batch size of **32**.

**Phase 1: Transfer Learning**
- Base model frozen; only top layers trained.
- **Epochs:** 10 | **Learning Rate:** 1e-3 | **Result:** Validation Accuracy ~88.87%

**Phase 2: Fine-Tuning**
- Unfroze top 30 layers of the EfficientNet base to permit granular feature adjustments.
- **Epochs:** 20 | **Learning Rate:** 1e-4 (Using `ReduceLROnPlateau` callback) | **Callbacks:** Early Stopping, Model Checkpoint.

### Final Results
- **Final Validation Accuracy:** `93.62%`
- **Validation Loss:** `0.3861`

The model yields robust diagonal accuracy across its Confusion Matrix, showcasing an excellent balance of Precision and Recall. *(Refer to `image.png` and `image-1.png` for graph visualizations, if placed in this directory).*

---

## 📦 Model Export

The optimized model weights are strictly saved on disk under the `.keras` format format. Code block syntax representation:

```python
model.save("plant_disease_model.keras")
```