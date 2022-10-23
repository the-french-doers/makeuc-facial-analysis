import numpy as np

from libs.mathUtils import get_distance_between_2d_points, calculate_error


class LandmarksManager:
    def __init__(self):
        self.silhouette = [
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
        ]
        self.nostrils = [[237, 19, 59, 74, 58, 165, 78, 238], [249, 289, 304, 288, 391, 308, 458, 457]]

    def get_sihouette_distance_matrix(self, landmarks, frame_shape) -> list:
        leftIris = landmarks[468]
        rightIris = landmarks[473]

        scale = get_distance_between_2d_points(leftIris, rightIris, frame_shape)

        result = []

        for i in self.silhouette:
            toAdd = []

            for j in self.silhouette:
                toAdd.append(get_distance_between_2d_points(landmarks[i], landmarks[j], frame_shape) / scale)

            result.append(toAdd)

        return result

    def get_nostril_distance_matrix(self, landmarks, frame_shape) -> list:
        leftIris = landmarks[468]
        rightIris = landmarks[473]

        scale = get_distance_between_2d_points(leftIris, rightIris, frame_shape)

        result = []

        for i_left, i_right in zip(self.nostrils[0], self.nostrils[1]):
            toAdd = []

            for j_left, j_right in zip(self.nostrils[0], self.nostrils[1]):
                left = get_distance_between_2d_points(landmarks[i_left], landmarks[j_left], frame_shape) / scale
                right = get_distance_between_2d_points(landmarks[i_right], landmarks[j_right], frame_shape) / scale

                toAdd.append((left + right) / 2)

            result.append(toAdd)

        return result

    def get_silhouette_change(self, prev_data_silhouette, data_silhouette) -> str:
        error = calculate_error(prev_data_silhouette, data_silhouette)

        if np.abs(error) <= 65:
            return "same"
        else:
            if error > 0:
                return "bigger"
            else:
                return "smaller"

    def get_nostril_change(self, prev_data_nostril, data_nostril) -> str:
        error = calculate_error(prev_data_nostril, data_nostril)

        if np.abs(error) <= 3:
            return "same"
        else:
            if error > 0:
                return "bigger"
            else:
                return "smaller"
