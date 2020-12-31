import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import math
from constants import *


class Eigen:

    def __init__(self, path):
        self.path = path
        self.m = len(os.listdir(self.path))
        self.imsize = 100
        self.norm_x = np.zeros((self.imsize * self.imsize, self.m))
        self.mean_face = np.zeros((self.imsize * self.imsize, 1))

    def label_extract(self):
        # Creating a labeled list for all the names
        name_list = [files for files in os.listdir(self.path)]
        # replacing numbered names with original names
        im_label = [nl.replace(nl, l) for nl in name_list for l in labels if l in nl]
        return im_label

    def image_processing(self, *args):
        # Loading images in a list
        images = [img.imread(self.path + os.sep + files) for files in os.listdir(self.path)]
        # plt.imshow(np.array(images[0]))

        # converting N*N to N^2
        column_vectors = [np.array(image).reshape((-1, 1)) for image in images]
        prev_column = column_vectors[0]

        # stacking the column vectors to one vector N^2 * m
        for count, _ in enumerate(column_vectors):
            if count + 1 == len(column_vectors):
                break
            stacked_vector = np.column_stack((prev_column, column_vectors[count + 1]))
            prev_column = stacked_vector
        if len(args):
            self.mean_face = args[0]
        else:
            average_face = np.mean(stacked_vector, axis=1)
            self.mean_face = average_face.reshape(10000, 1)
        # Display Average face
        # plt.imshow(average_face.reshape((100, 100)))
        # plt.show()

        # mean normalization
        self.norm_x = stacked_vector - self.mean_face
        return stacked_vector, self.mean_face

    def covariance(self):
        covariance_mat = np.dot(self.norm_x.T, self.norm_x) / self.m - 1
        return covariance_mat

    def eigen_value(self):
        eig_value, eig_vec = np.linalg.eig(self.covariance())
        # Convert lower dimension eigen vector to original dimension ui= A*vi(eigen vector relation between A.T*A and
        # A*A.T)
        eig_face = np.dot(self.norm_x, eig_vec)
        # Normalize eigenvectors
        eig_face = eig_face / np.linalg.norm(eig_face, axis=0)
        # self.display_data(eig_face, 'Eigenface')
        return eig_value, eig_face

    def weights_calculation(self, eig_face, mean_face):
        weights = np.dot(self.norm_x.T, eig_face)
        reconstruction = mean_face + np.dot(eig_face, weights.T)
        return weights, reconstruction

    @staticmethod
    def display_data(data, title):
        shape = int(math.sqrt(data.shape[1]))
        figure, axes = plt.subplots(shape, shape)
        k = 0
        for i in range(shape):
            for j in range(shape):
                axes[i, j].imshow(data[:, k].reshape((100, 100)))
                k += 1
        plt.title(title, loc='left')
        plt.show()
