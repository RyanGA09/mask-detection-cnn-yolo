# YOLO Face Mask Detection 😷

Real-time face mask detection using YOLOv5. This model classifies whether a person is wearing a mask correctly, not wearing a mask, or wearing it incorrectly.

## Dataset
[Kaggle - Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

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

## Training
```bash
python train.py --img 416 --batch 16 --epochs 50 --data data/face_mask_data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt
```

## Inference
```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.4 --source data/images/val
```

## Example Result
![example](runs/detect/exp/image1.jpg)
