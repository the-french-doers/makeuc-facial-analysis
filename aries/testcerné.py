from urllib.error import HTTPError
import cv2
import mediapipe as mp
import numpy as np

from libs.mpFace import MpFace
from libs.dbManager import DbManager
from libs.landmarksManager import LandmarksManager

face = MpFace()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255))

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)

    try:
        landmarks_list, landmarks = face.get_landmarks(frame)
        
        # right rectangle
        start_point = (int(landmarks[341].x*frame.shape[1])
                    ,int(landmarks[341].y*frame.shape[0]))
        end_point = (int(landmarks[346].x*frame.shape[1])
                    ,int(landmarks[346].y*frame.shape[0]))
        
        # non puffy rectangle
        start_point2 = (int(landmarks[329].x*frame.shape[1])
                        ,int(landmarks[329].y*frame.shape[0]))
        end_point2 = (int(landmarks[266].x*frame.shape[1])
                    ,int(landmarks[266].y*frame.shape[0]))
        
        # special info
        color = (0,0,255)
        thickness = 2
        
        # cernes ou pas 
        
        # zone supposée cernée
        
        cerne_zone1 = frame[int(landmarks[341].y*frame.shape[0]):int(landmarks[346].y*frame.shape[0]),
                           int(landmarks[341].x*frame.shape[1]):int(landmarks[346].x*frame.shape[1])]
        
        avg_colors1 = cerne_zone1.mean(axis=0).mean(axis=0)
        
        
        # zone supposée non cernée
        cerne_zone2 = frame[int(landmarks[329].y*frame.shape[0]):int(landmarks[266].y*frame.shape[0]),
                           int(landmarks[329].x*frame.shape[1]):int(landmarks[266].x*frame.shape[1])]
        
        avg_colors2 = cerne_zone2.mean(axis=0).mean(axis=0)
        
        sum1 = sum(avg_colors1)
        sum2 = sum(avg_colors2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
  
        org = (50, 50)
        
        fontScale = 1

        colorbis = (255, 0, 0)
        
        
        
        if sum1 and sum2 and sum1 < sum2 : 
            cv2.putText(frame, 'Cernes', org, font, 
                            fontScale, colorbis, thickness, cv2.LINE_AA)
        elif sum1 and sum2: 
            cv2.putText(frame, 'pas de Cernes', org, font, 
                            fontScale, colorbis, thickness, cv2.LINE_AA)
            
            
       
        
        cv2.rectangle(frame, start_point2, end_point2, color, thickness)

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        cv2.rectangle(frame, start_point, end_point, color, thickness)

        

        mp_drawing.draw_landmarks(image=frame, landmark_list=landmarks_list, landmark_drawing_spec=drawing_spec)
    except HTTPError as err:
        print(err)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()





        
















    

