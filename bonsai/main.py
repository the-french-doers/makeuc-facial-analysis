import cv2
import mediapipe as mp

from libs.mpFace import MpFace
from libs.dbManager import DbManager
from libs.landmarksManager import LandmarksManager

face = MpFace()
dbManager = DbManager("db")
landmarksManager = LandmarksManager()

data = dbManager.loadFile("data.json")
prev_data = dbManager.loadFile("data.json")

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

    try:
        landmarks_list, landmarks = face.get_landmarks(frame)

        mp_drawing.draw_landmarks(image=frame, landmark_list=landmarks_list, landmark_drawing_spec=drawing_spec)

        silhouette_distance_matrix = landmarksManager.get_sihouette_distance_matrix(landmarks, frame.shape)
        
        nostril_distance_matrix = landmarksManager.get_nostril_distance_matrix(landmarks, frame.shape)

        data["silhouette"] = silhouette_distance_matrix
        data["nostril"] = nostril_distance_matrix

        silhouette_change = landmarksManager.get_silhouette_change(prev_data["silhouette"], data["silhouette"])

        nostril_change = landmarksManager.get_nostril_change(prev_data["nostril"], data["nostril"])

        cv2.putText(
            frame,
            f"Silhouette : {silhouette_change}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame, f"Nostril : {nostril_change}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
        )
    except Exception as err:
        print(err)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        dbManager.writeFile("data.json", data)
        break

cap.release()
