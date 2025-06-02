# YOLO Face Mask Detection 😷

Real-time face mask detection using YOLOv5. This model classifies whether a person is wearing a mask correctly, not wearing a mask, or wearing it incorrectly.

## Dataset

[Kaggle - Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

## Directory Structure

```bash
face-mask-detection/
├── src/
│   ├── dataset_loader.py
│   ├── model.py
│   └── train_eval.py
├── notebooks/
│   └── face_mask_detection.ipynb
├── app/
│   └── webcam_inference.py
├── models/
├── data/
├── visualizations/
├── README.md
└── requirements.txt

```

- `notebooks/`: Notebook training dan evaluasi
- `src/`: Script model dan data loader
- `data/`: Dataset dari Kaggle
- `models/`: Model hasil training
- `visualizations/`: Grafik training dan evaluasi
- `app/`: Aplikasi webcam deteksi masker

## Classes

- with_mask
- without_mask
- mask_weared_incorrect

## Installation

```bash
git clone https://github.com/yourname/yolo-face-mask-detection
cd yolov5
pip install -r requirements.txt
```

## Usage Steps

1. Simpan `kaggle.json` Anda di luar folder proyek ini
2. Install dependensi: `pip install -r requirements.txt`
3. Jalankan `notebooks/face_mask_detection.ipynb` untuk training
4. Jalankan `app/webcam_inference.py` untuk aplikasi webcam

## 📥 Preparation Steps

1. Install library:

   ```bash
   pip install -r requirements.txt

   ```

2. Save kaggle.json outside the project folder
3. Download the dataset from Kaggle using `src/dataset_loader.py`:

   ```bash
   from src.dataset_loader import download_dataset
   download_dataset("C:/Users/YourName/kaggle.json")

   ```
