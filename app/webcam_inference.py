import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('models/mask_detector.h5')
labels = ['with_mask', 'without_mask', 'mask_weared_incorrect']

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    face = cv2.resize(frame, (224, 224))
    face = np.expand_dims(face, axis=0) / 255.0
    pred = model.predict(face)
    label = labels[np.argmax(pred)]

    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()