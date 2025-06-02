import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadStreams
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
import cv2

weights = 'runs/train/exp/weights/best.pt'
imgsz = 416

device = select_device('')
model = DetectMultiBackend(weights, device=device)
stride, names = model.stride, model.names

dataset = LoadStreams(0, img_size=imgsz, stride=stride)
for path, img, im0s, vid_cap, s in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    for i, det in enumerate(pred):
        im0 = im0s[i].copy()
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)
                cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow('YOLOv5 Webcam - Face Mask Detection', im0)
        if cv2.waitKey(1) == ord('q'):
            break
