import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------
# Load Saved Model
# -------------------------
# Assuming the model is in your Google Drive after mounting.
# Please update 'path_to_model_in_drive' to the exact folder containing mask_model.keras
model_path = "mask_model.keras"
model = load_model(model_path)

# Load Haarcascade for face detection
# You might need to download 'haarcascade_frontalface_default.xml' if it's not present
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam (this part won't work directly in Colab without a connected webcam and display)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop the face
        face = frame[y:y+h, x:x+w]

        # Preprocess same as training
        face = cv2.resize(face, (150, 150))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        # Prediction
        pred = model.predict(face)[0][0]

        # Assign label
        if pred < 0.5:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        # Draw rectangle and label on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # This imshow will likely not work in typical Colab environments unless configured for display
    cv2.imshow("Mask Detection - Press Q to Exit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()