import cv2
import mediapipe as mp

from libs.mpFace import MpFace

face = MpFace()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)

    landmarks_list, landmarks = face.get_landmarks(frame)

    mp_drawing.draw_landmarks(image=frame, landmark_list=landmarks_list, landmark_drawing_spec=drawing_spec)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
