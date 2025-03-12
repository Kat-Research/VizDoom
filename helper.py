import cv2
import numpy as np
from gymnasium.spaces import Box
from matplotlib import pyplot as plt

def rgb_to_greyscale(observation: np.ndarray) -> np.ndarray:
    if len(observation.shape) != 3:
        raise ValueError(f"tried to greyscale image thats not rgb {observation.shape}")

    return cv2.cvtColor(
        np.moveaxis(observation, 0, -1) if observation.shape[0] == 3 else observation,
        cv2.COLOR_RGB2GRAY
    )

def display_grey_scale(grey_scale_observation: np.ndarray) -> None:
    plt.imshow(cv2.cvtColor(grey_scale_observation, cv2.COLOR_BGR2RGB))