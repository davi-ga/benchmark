import numpy as np


def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)


def scaled_sigmoid(x, max_score=20, stretch=10, center=0):
    stretch = stretch / 5
    x = (x + center) / stretch
    return round(sigmoid(x) * max_score, 2)
