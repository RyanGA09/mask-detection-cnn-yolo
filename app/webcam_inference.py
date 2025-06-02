import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Path folder script ini
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path folder model DNN face detector
dnn_folder = os.path.join(script_dir, '..', 'models', 'dnn')
proto_path = os.path.join(dnn_folder, 'deploy.prototxt')
model_file = os.path.join(dnn_folder, 'res10_300x300_ssd_iter_140000.caffemodel')

# Cek apakah file model DNN ada
if not os.path.exists(proto_path) or not os.path.exists(model_file):
    raise FileNotFoundError(f"File model DNN face detector tidak ditemukan di:\n{proto_path}\n{model_file}\n"
                            f"Silakan download manual dan simpan di folder tersebut.")

# Load model DNN face detector
face_net = cv2.dnn.readNetFromCaffe(proto_path, model_file)

# Path model mask detector
mask_model_path = os.path.join(script_dir, '..', 'models', 'mask_detector.h5')
if not os.path.exists(mask_model_path):
    raise FileNotFoundError(f"Model mask detector tidak ditemukan di path: {mask_model_path}")

print("Loading model mask detector dari:", mask_model_path)
mask_model = load_model(mask_model_path)

labels = ['with_mask', 'without_mask', 'mask_weared_incorrect']

# Mulai capture webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]

    # Prepare input blob untuk face detector DNN
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locations = []

    # Iterasi deteksi wajah
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # Pastikan kotak dalam frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (224, 224))
            face_norm = face_resized.astype('float32') / 255.0
            faces.append(face_norm)
            locations.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces_np = np.array(faces)
        preds = mask_model.predict(faces_np)

        for pred, (startX, startY, endX, endY) in zip(preds, locations):
            label_idx = np.argmax(pred)
            label = labels[label_idx]
            confidence = pred[label_idx]

            # Warna kotak berdasarkan label
            if label == 'with_mask':
                color = (0, 255, 0)  # hijau
            elif label == 'without_mask':
                color = (0, 0, 255)  # merah
            else:
                color = (0, 255, 255)  # kuning untuk 'mask_weared_incorrect'

            label_text = f"{label}: {confidence*100:.1f}%"
            cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow('Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
