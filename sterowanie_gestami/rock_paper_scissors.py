import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    if index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y and thumb_tip.y > index_tip.y:
        return "Rock"
    elif index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y:
        return "Paper"
    else:
        return "Scissors"

def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "Draw"
    elif (user_choice == "Rock" and computer_choice == "Scissors") or \
         (user_choice == "Paper" and computer_choice == "Rock") or \
         (user_choice == "Scissors" and computer_choice == "Paper"):
        return "You Win!"
    else:
        return "Computer Wins!"

cap = cv2.VideoCapture(0)
start_time = time.time()
countdown = 5  
game_state = "countdown"  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    current_time = time.time()
    elapsed_time = current_time - start_time

    if game_state == "countdown":
        remaining_time = countdown - int(elapsed_time)
        cv2.putText(frame, f'Time left: {remaining_time}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if remaining_time <= 0:
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark
                    user_choice = recognize_gesture(landmarks)
                    break
            else:
                user_choice = "No Gesture"

            computer_choice = get_computer_choice()
            result_text = determine_winner(user_choice, computer_choice)
            game_state = "show_result"
            start_time = time.time()  


    elif game_state == "show_result":
        cv2.putText(frame, f'You: {user_choice}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Computer: {computer_choice}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, result_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if elapsed_time >= 3: 
            game_state = "countdown"
            start_time = time.time()  

    cv2.imshow('Rock Paper Scissors', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
