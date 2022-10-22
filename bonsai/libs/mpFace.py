import cv2


class MpFace:
    def __init__(self):
        import mediapipe as mp

        self.mp_face = mp.solutions.face_mesh.FaceMesh()

    def get_landmarks(self, frame):
        frame.flags.writeable = False

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.mp_face.process(frame)

        frame.flags.writeable = True

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        landmarks_list, landmarks = [], []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_list = face_landmarks
                landmarks = face_landmarks.landmark

        return landmarks_list, landmarks
