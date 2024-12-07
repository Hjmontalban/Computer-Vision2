import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained face detection model and the face recognition model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the face recognition model with custom handling for DepthwiseConv2D
custom_objects = {'DepthwiseConv2D': DepthwiseConv2D}
model = load_model('keras_model.h5', custom_objects=custom_objects)  # Replace with your model's path

# Load labels from CSV file
labels_df = pd.read_csv('face_recognition_labels.csv')  # Replace with your CSV file's path
labels = labels_df['label'].tolist()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Preprocess the face for recognition
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))  # Resize to the input size of the model
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        # Predict the identity of the face
        preds = model.predict(face)[0]
        j = np.argmax(preds)
        label = labels[j]

        # Display the label and confidence on the frame
        label_text = f"{label}: {preds[j] * 100:.2f}%"
        cv2.putText(frame, label_text, (x, y - 10), font, 0.45, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
