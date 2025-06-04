# YOLO Face Mask Detection 😷

Real-time face mask detection using YOLOv5. This model classifies whether a person is wearing a mask correctly, not wearing a mask, or wearing it incorrectly.

## Dataset

[Kaggle - Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

## Directory Structure

```bash
face-mask-detection/
├── module/
│   ├── __init__.py
│   ├── dataset_loader.py
│   ├── model.py
│   └── train_eval.py
├── notebooks/
│   └── face_mask_detection.ipynb
├── app/
│   └── webcam_inference.py
├── models/
│   └── dnn/
├── data/
├── visualizations/
├── README.md
└── requirements.txt

```

- `notebooks/`: Notebook training and evaluation
- `src/`: Script model and data loader
- `data/`: Dataset from Kaggle
- `models/`: Result training model
- `visualizations/`: Training and evaluation graph
- `app/`: Mask detection webcam app

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

1. Save your `kaggle.json` outside this project folder
2. Install the dependencies: `pip install -r requirements.txt`
3. Run `notebooks/face_mask_detection.ipynb` for training
4. Run `app/webcam_inference.py` for webcam application 4.

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
