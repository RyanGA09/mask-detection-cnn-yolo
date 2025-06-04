# Face Mask Detection using CNN (MobileNetV2) and YOLOv5 😷

Real-time face mask detection using YOLOv5. This model classifies whether a person is:

- ✅ Wearing a mask correctly (`with_mask`)
- ❌ Not wearing a mask (`without_mask`)
- ⚠️ Wearing a mask incorrectly (`mask_weared_incorrect`)

## 📁 Dataset

[Kaggle - Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

## 📂 Directory Structure

```bash
face-mask-detection/
├── module/
│   ├── __init__.py
│   ├── dataset_loader.py
│   ├── model.py
│   └── train_eval.py
├── notebook/
│   └── face_mask_detection.ipynb
├── app/
│   └── webcam_inference.py
├── models/
│   ├── dnn/
│   ├── mask_detector.h5 # CNN model training results
│   └── best_mask_detector.h5 # CNN model with best validation accuracy
├── data/
├── test/
├── visualizations/
│   └── predictions.csv # Prediction results y_true and y_pred
├── README.md
└── requirements.txt

```

- `notebooks/`: Notebook training and evaluation
- `module/`: Script model and data loader
- `data/`: Dataset from Kaggle
- `test/`: sample data test
- `models/`: Result training model
- `visualizations/`: Training and evaluation graph
- `app/`: Mask detection webcam app
- `requirements.txt`: Python Dependency

## 📊 Classes

- `with_mask`
- `without_mask`
- `mask_weared_incorrect`

## 🔗 Dependencies / Requirements

Make sure you have the following libraries:

- tensorflow
- opencv-python
- matplotlib
- seaborn
- numpy
- scikit-learn
- kaggle
- torch
- yolov5

## Installation

```bash
git clone https://github.com/RyanGA09/yolo-face-mask-detection
cd yolo-face-mask-detection
pip install -r requirements.txt
```

## 🧠 CNN Model Architecture

### 🧬 CNN (MobileNetV2)

- Pre-trained MobileNetV2 from ImageNet
- Top layer customisation:
  - Global Average Pooling
  - Dropout
  - Dense Layer (3 softmax outputs)
- Data augmentation and validation using `ImageDataGenerator`
- Early Stopping and Model Checkpoint during training

### 🎯 YOLOv5

- Using pretrained YOLOv5s from Ultralytics
- Used for visual inference of sample images

## ▶️ Usage Steps

1. Save your `kaggle.json` outside this project folder
2. Install the dependencies: `pip install -r requirements.txt`
3. Run `notebook/face_mask_detection.ipynb` for training
4. Download & Extract Dataset from Kaggle
5. Organise Dataset into folders based on class labels
6. Image data Augmentation & Preprocessing
7. Training CNN (MobileNetV2) for image classification
8. Model evaluation with:
   - Accuracy & Loss plot
   - Confusion Matrix (normal & raw)
   - Classification report
9. Save prediction results to CSV file
10. Prediction on one image
11. Inference using YOLOv5 on random images from each category
12. Run `app/webcam_inference.py` for webcam application 4.

## 📈 Evaluation Metrics

After training, the model will be evaluated using:

- Accuracy & Loss Plot (Training vs Validation)
- Confusion Matrix (normal and normalised)
- Classification Report (`Precision`, `Recall`, `F1-Score`)

Evaluation results are saved in a folder `visualizations/`.

## 🖼️ Prediksi dan Inferensi

- Prediksi dapat dilakukan pada satu gambar menggunakan model CNN
- Inferensi deteksi objek menggunakan YOLOv5 pretrained model
  - Menampilkan bounding box dan label langsung di gambar

## 📥 Preparation Steps

1. Install library:

   ```bash
   pip install -r requirements.txt

   ```

2. Save kaggle.json outside the project folder
3. Download the dataset from Kaggle using `module/dataset_loader.py`:

   ```bash
   from module import download_dataset
   download_dataset(kaggle_json_path)

   ```

4. Preprocess the data:

   ```bash
   from module import organize_dataset
   organize_dataset()

   ```

5. Train and evaluate model from notebook

## 👨‍💻 Author

Ryan Gading Abdullah

- [GitHub](https://github.com/RyanGA09)
- [Instagram](https://instagram.com/ryan_g._a)
- [Linkedin](https://www.linkedin.com/in/ryan-gading-abdullah/)

## 🪪 LICENSE

Copyright &copy; 2025 Ryan Gading Abdullah. All rights reserved.

This project is licensed under the MIT License - see the [MIT LICENSE](LICENSE) for details.
