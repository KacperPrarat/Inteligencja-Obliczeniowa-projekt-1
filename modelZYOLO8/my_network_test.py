import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
net = cv2.dnn.readNet("last.pt", "yolo_config.cfg")

model = YOLO("last.pt")  # load a custom model


# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Capture live video
cap = cv2.VideoCapture(0)  # Change the argument to the video file path if not using webcam

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame (if necessary)
    # e.g., resizing, normalization

    # Perform inference
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Post-process detection
    # Extract keypoints and process as needed

    # Visualize results
    # Draw keypoints on the frame

    # Display the resulting frame
    cv2.imshow('Key Point Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
