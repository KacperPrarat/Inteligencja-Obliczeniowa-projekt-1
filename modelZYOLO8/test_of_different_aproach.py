import cv2
import pyautogui
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

# model = YOLO("yolov8m-pose.pt")
model = YOLO("last.pt")

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


## Setup mediapipe instance
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        #cv2.putText(annotated_frame, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        #writer.append_data(annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    # # Recolor image to RGB
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # image.flags.writeable = False
      
    #     # Make detection
    # results = pose.process(image)
    
    #    # Recolor back to BGR
    # image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    #     # Extract landmarks
    # try:
    #     landmarks = results.pose_landmarks.landmark
            
    #         # Get coordinates
    #     shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    #     elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    #     wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
    #         # Calculate angle
    #     angle = calculate_angle(shoulder, elbow, wrist)
            
    #         # Visualize angle
    #     cv2.putText(image, str(angle), 
    #                 tuple(np.multiply(elbow, [640, 480]).astype(int)), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                # )
            
    # except:
        # pass
        
    #     # Render curl counter
    #     # Setup status box
    # cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
    #     # Rep data
    # cv2.putText(image, 'REPS', (15,12), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    # cv2.putText(image, str(counter), 
    #                 (10,60), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
    #     # Stage data
    # cv2.putText(image, 'STAGE', (65,12), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    # cv2.putText(image, stage, 
    #                 (60,60), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
    #     # Render detections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
    #                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
    #                              )               
        
    # cv2.imshow('Mediapipe Feed', image)

    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break

cap.release()
cv2.destroyAllWindows()
