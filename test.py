import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os


class Test:
    def __init__(self, train_stacked_images, train_weights, test_weights):
        self.train_weights = train_weights
        self.test_weights = test_weights
        self.original_images = train_stacked_images

    # matching the test weights with train weights with lowest distance
    def match_index(self, train_label):
        predicted_label =[]
        for i in range(self.test_weights.shape[0]):
            min = self.min_distance(self.train_weights, self.test_weights[i, :])
            predicted_label.append(train_label[min])
        return predicted_label

    @staticmethod
    def min_distance(train_weights, test_weight):
        distances_dict = {i: np.linalg.norm(abs(test_weight - train_weights[i, :]))
                          for i in range(train_weights.shape[0])}
        temp = min(distances_dict.values())
        k_temp = [key for key in distances_dict if distances_dict[key] == temp]
        return k_temp[0]
