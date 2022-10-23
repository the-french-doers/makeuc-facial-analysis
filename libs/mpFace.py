import cv2


class MpFace:
    def __init__(self) -> None:
        import mediapipe as mp

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def get_landmarks(self, frame) -> list:
        frame.flags.writeable = False

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.mp_face.process(frame)

        landmarks_list, landmarks = [], []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_list = face_landmarks
                landmarks = face_landmarks.landmark

        return landmarks_list, landmarks
