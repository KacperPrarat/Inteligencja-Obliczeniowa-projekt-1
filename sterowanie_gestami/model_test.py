import cv2
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Load your pre-trained model with .keras extension
model = tf.keras.models.load_model('game-model.keras')

# Print model summary to understand the architecture
model.summary()

# Define a function to preprocess the image
def preprocess_frame(frame, target_size=(300, 200)):
    img = cv2.resize(frame, target_size)
    img_array = img.astype(np.float32) / 255.0  # Normalize the image array
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to predict the gesture
def predict_gesture(model, frame):
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    predicted_class = np.argmax(predictions, axis=1)

    class_names = ['rock', 'paper', 'scissors']
    return class_names[predicted_class[0]]

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict gesture
    predicted_gesture = predict_gesture(model, frame)

    # Display the predicted gesture on the frame
    cv2.putText(frame, f'Prediction: {predicted_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame with the prediction
    cv2.imshow('Rock Paper Scissors Gesture Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
