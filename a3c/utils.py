import os
import numpy as np


def to_list(value):
    if value is None:
        return None
    elif isinstance(value, tuple):
        value = list(value)
    elif not isinstance(value, list):
        value = [value]
    return value


def rgb2y(image, scalars=[0.299, 0.587, 0.114]):
    y = np.zeros(image.shape[:2], dtype=np.float32)
    for idx, scalar in enumerate(scalars):
        y += scalar * image[:, :, idx]
    return y


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
