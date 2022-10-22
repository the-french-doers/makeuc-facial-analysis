import numpy as np


def get_distance_between_2d_points(p1, p2, shape) -> float:
    return ((p1.x * shape[1] - p2.x * shape[1]) ** 2 + (p1.y * shape[1] - p2.y * shape[1]) ** 2) ** 0.5


def calculate_error(m1, m2) -> float:
    return np.sum(np.array(m1) - np.array(m2))
