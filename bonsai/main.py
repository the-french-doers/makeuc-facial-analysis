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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255))

BPMS = 0
ITERATIONS = 0

ARR = []
ARR_time = []


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

        nostril_change = landmarksManager.get_nostril_change(prev_data["nostril"], data["nostril"])

        left_upper_corner = (int(landmarks[109].x * frame.shape[1]), int(landmarks[109].y * frame.shape[0]))

        right_lower_corner = (int(landmarks[285].x * frame.shape[1]), int(landmarks[285].y * frame.shape[0]))

        detection_frame = frame[
            left_upper_corner[1] : right_lower_corner[1],
            left_upper_corner[0] : right_lower_corner[0],
        ]

        detection_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)

        detection_frame[:, :, 0] = detection_frame[:, :, 0] / 1000

        print(detection_frame)

        avg = np.where((detection_frame[:, :, 0] <= 0.1))

        avgg = detection_frame[avg]

        average_color = avgg.mean(axis=0).mean(axis=0)
        detection_frame[avg] = average_color

        d_img = np.ones((312, 312, 3), dtype=np.uint8)
        d_img[:, :] = average_color

        cv2.imshow("detection", d_img)

        if len(ARR) >= 30 * 10:
            computed_ARR_time = [
                (ARR_time[index] - ARR_time[0]) if index != 0 else 0 for index, _ in enumerate(ARR_time)
            ]

            plt.plot(computed_ARR_time, ARR)
            plt.show()

            fftData = np.fft.fft(ARR)
            freq = np.fft.fftfreq(300, 1 / 30)
            fftData = np.fft.fftshift(fftData)
            freq = np.fft.fftshift(freq)

            plt.xlim(0, 4)
            plt.ylim(0, 100)

            plt.plot(freq, np.abs(fftData))
            plt.show()

            ARR.pop(0)
            ARR_time.pop(0)

        ARR.append(average_color)
        ARR_time.append(time())

        bpm = round(gaussManager.getBpm(detection_frame))

        BPMS += bpm if bpm else 0
        ITERATIONS += 1 if bpm else 0

        mp_drawing.draw_landmarks(image=frame, landmark_list=landmarks_list, landmark_drawing_spec=drawing_spec)

        cv2.circle(frame, left_upper_corner, 5, (0, 0, 255), -1)
        cv2.circle(frame, right_lower_corner, 5, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"Silhouette : {silhouette_change}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Nostril : {nostril_change}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Bpm : {bpm if bpm else 'Waiting...'}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        if ITERATIONS >= cap.get(cv2.CAP_PROP_FPS) * 10:
            cv2.putText(
                frame,
                f"Avg Bpm (10 sec) : {round(BPMS / ITERATIONS)}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    except HTTPError as e:
        print(f"Error : {e}")

    cv2.imshow("Frame", frame)
    cv2.imshow("Detection", detection_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        dbManager.writeFile("data.json", data)
        break

cap.release()
