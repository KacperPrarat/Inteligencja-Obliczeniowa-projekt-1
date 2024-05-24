import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)
keyboard = Controller()

prev_x, prev_y = None, None

MOVE_THRESHOLD = 10

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image = cv2.flip(image, 1)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            x = int(hand_landmarks.landmark[8].x * image.shape[1])
            y = int(hand_landmarks.landmark[8].y * image.shape[0])
            
            if prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy = y - prev_y
                
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
            
            prev_x, prev_y = x, y

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
