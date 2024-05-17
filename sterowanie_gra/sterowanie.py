import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

# Minecraft control variables
threshold_distance = 0.1   # Adjust as needed
jump_threshold = 150  # Adjust as needed
turn_threshold = 0.1   # Adjust as needed
head_turn_threshold = 0.08  # Adjust as needed
jump_pressed = False
turn_direction = 0     # -1 for left, 1 for right, 0 for no turn
knee_hip_threshold = 0.02  # Adjust as needed

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# Function to control Minecraft based on detected gestures
def control_minecraft(landmarks):
    global jump_pressed, turn_direction


    
    # Get landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)


   
    # Jumping control
    if results.pose_landmarks:
        hip_landmarks = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]]
        knee_landmarks = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]]

        knees_above_hips = [knee.y < hip.y - knee_hip_threshold for knee,
                            hip in zip(knee_landmarks, hip_landmarks)]

        if any(knees_above_hips):  # If at least one knee is above the hips
            pyautogui.keyDown('w')  # Press 'w' to move forward
            time.sleep(2)
            print("Walking")
        else:
            pyautogui.keyUp('w')  # Release 'w' key

        # if all(knees_above_hips):  # If both knees are above the hips
        #     pyautogui.press('space')  # Press 'space' to jump
        #     print("Jumping")
    
    # Turn control
    if left_shoulder.y > right_shoulder.y + threshold_distance:
        if turn_direction != -1:
            pyautogui.keyDown('a')
            print("a")
            turn_direction = -1
    elif right_shoulder.y > left_shoulder.y + threshold_distance:
        if turn_direction != 1:
            pyautogui.keyDown('d')
            print("d")
            turn_direction = 1
    else:
        if turn_direction != 0:
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')
            turn_direction = 0
    
    # Mouse control based on head rotation
    if nose.x < 0.5 - head_turn_threshold:
        pyautogui.moveRel(10, 0)  # Move mouse cursor left by 10 pixels
        print("mouse to the right")
    elif nose.x > 0.5 + head_turn_threshold:
        pyautogui.moveRel(-10, 0)   # Move mouse cursor right by 10 pixels
        print("mouse to the left")
    
    # Mouse buttons control 

    if right_arm_angle < 130:
        pyautogui.click(button="right")
        print("Right mouse button clicked")
    if left_arm_angle < 130:
        pyautogui.click(button="left")
        print("Left mouse button clicked")
    

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make detection
        results = pose.process(image)
    
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Control Minecraft based on detected gestures
            control_minecraft(landmarks)
            
            # Draw keypoints on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                    ) 
                       
        except:
            pass
        
        cv2.imshow('Mediapipe Feed', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
