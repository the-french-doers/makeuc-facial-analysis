from time import time
from urllib.error import HTTPError
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

from libs.mpFace import MpFace
from libs.dbManager import DbManager
from libs.landmarksManager import LandmarksManager
from libs.gaussManager import GaussManager

cap = cv2.VideoCapture(0)

face = MpFace()
dbManager = DbManager("db")
landmarksManager = LandmarksManager()
gaussManager = GaussManager()

data = dbManager.loadFile("data.json")
prev_data = dbManager.loadFile("data.json")
questions = dbManager.loadFile("questions.json")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255))

SILHOUETTE_STATE = []
NOSTRIL_STATE = []
HAS_DARK_CIRCLES = 0
BPMS = 0
ITERATIONS = 0

print("=========================================")
print(
    "Hello, I will be your personal assistant throughout your treatment. In order to monitor your health and to be able to help you as best as possible, we need you to answer a few questions before taking part in a short analysis of your vital functions."
)
print("")

answers = []

for question in questions:
    answer = str(input(f"{question} [Y(yes) / N(o)] : "))
    answers.append(answer)

dbManager.writeFile("answers.json", answers)

print("")
print("=========================================")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)

    try:
        landmarks_list, landmarks = face.get_landmarks(frame)

        silhouette_distance_matrix = landmarksManager.get_sihouette_distance_matrix(landmarks, frame.shape)

        nostril_distance_matrix = landmarksManager.get_nostril_distance_matrix(landmarks, frame.shape)

        data["silhouette"] = silhouette_distance_matrix
        data["nostril"] = nostril_distance_matrix

        silhouette_change = landmarksManager.get_silhouette_change(prev_data["silhouette"], data["silhouette"])

        SILHOUETTE_STATE.append(silhouette_change)

        nostril_change = landmarksManager.get_nostril_change(prev_data["nostril"], data["nostril"])

        NOSTRIL_STATE.append(nostril_change)

        has_dark_circles = landmarksManager.has_dark_circles(landmarks, frame.shape, frame)

        if has_dark_circles:
            HAS_DARK_CIRCLES += 1

        left_upper_corner = (int(landmarks[109].x * frame.shape[1]), int(landmarks[109].y * frame.shape[0]))

        right_lower_corner = (int(landmarks[285].x * frame.shape[1]), int(landmarks[285].y * frame.shape[0]))

        detection_frame = frame[
            left_upper_corner[1] : right_lower_corner[1],
            left_upper_corner[0] : right_lower_corner[0],
        ]

        bpm = round(gaussManager.getBpm(detection_frame))

        BPMS += bpm if bpm else 0
        ITERATIONS += 1 if bpm else 0

        mp_drawing.draw_landmarks(image=frame, landmark_list=landmarks_list, landmark_drawing_spec=drawing_spec)

        cv2.putText(
            frame,
            f"Actual silhouette VS previous silhouette : {silhouette_change}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Actual nostril VS previous nostril : {nostril_change}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Has dark circles : {has_dark_circles}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Heart beat per minute : {bpm if bpm else 'Waiting...'}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Respiration per minute  : {bpm / 4 if bpm else 'Waiting...'}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        if ITERATIONS >= cap.get(cv2.CAP_PROP_FPS) * 10:
            data["bpm"] = round(BPMS / ITERATIONS)
            data["rpm"] = round(BPMS / ITERATIONS / 4)

            cv2.putText(
                frame,
                f"Avg Bpm (10 sec) : {round(BPMS / ITERATIONS)}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Avg Rpm (10 sec) : {round(BPMS / ITERATIONS / 4)}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"Please keep a neutral attitude and keep your head steady.",
            (10, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Press 'q' to end the analysis and save your data.",
            (10, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    except Exception as e:
        print(f"Error : {e}")

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        data["silhouette_state"] = max(set(SILHOUETTE_STATE), key=SILHOUETTE_STATE.count)
        data["nostril_state"] = max(set(NOSTRIL_STATE), key=NOSTRIL_STATE.count)

        if HAS_DARK_CIRCLES / ITERATIONS > 0.5:
            data["has_dark_circles"] = True

        dbManager.writeFile("data.json", data)
        break

print("")
print("Everything was successfully saved !")
print("")

cap.release()
