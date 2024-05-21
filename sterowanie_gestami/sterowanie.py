import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)
keyboard = Controller()

# Variables to store the previous coordinates of the index finger
prev_x, prev_y = None, None

# Define movement thresholds
MOVE_THRESHOLD = 10

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Flip the image horizontally for natural interaction
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(image)
    
    # Convert the image color back so it can be displayed
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the coordinates of the tip of the index finger (landmark 8)
            x = int(hand_landmarks.landmark[8].x * image.shape[1])
            y = int(hand_landmarks.landmark[8].y * image.shape[0])
            
            # Check if previous coordinates exist
            if prev_x is not None and prev_y is not None:
                # Calculate the differences
                dx = x - prev_x
                dy = y - prev_y
                
                # Determine the direction of the movement
                if abs(dx) > abs(dy) and abs(dx) > MOVE_THRESHOLD:
                    if dx > 0:
                        print("Right")
                        keyboard.press(Key.right)
                        keyboard.release(Key.right)
                    else:
                        print("Left")
                        keyboard.press(Key.left)
                        keyboard.release(Key.left)
                elif abs(dy) > abs(dx) and abs(dy) > MOVE_THRESHOLD:
                    if dy > 0:
                        print("Down")
                        keyboard.press(Key.down)
                        keyboard.release(Key.down)
                    else:
                        print("Up")
                        keyboard.press(Key.up)
                        keyboard.release(Key.up)
            
            # Update the previous coordinates
            prev_x, prev_y = x, y

    # Display the resulting frame
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
