import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os


class Test:
    def __init__(self, train_stacked_images, train_weights, test_weights):
        self.train_weights = train_weights
        self.test_weights = test_weights
        self.original_images = train_stacked_images

    def man_distance(self):
        dict_norm = {i: np.linalg.norm(abs(self.test_weights[i, :] - self.train_weights))
                     for i in range(self.test_weights.shape[0])}
        return dict_norm
