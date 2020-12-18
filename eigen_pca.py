import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os


class Eigen:

    def __init__(self, path):
        self.path = path
        self.m = len(os.listdir(self.path))
        self.imsize = 100
        self.norm_x = np.zeros((self.imsize*self.imsize, self.m))
        self.mean_face = np.zeros((self.imsize*self.imsize, 1))

    def image_processing(self):
        # Loading images in a list
        images = [img.imread('./cropped_image' + os.sep + files) for files in os.listdir('./cropped_image')]
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
        # Display faces
        # self.display_data(stacked_vector,'original image')

        average_face = np.mean(stacked_vector, axis=1)
        self.mean_face = average_face.reshape(10000, 1)
        # Display Average face
        # plt.imshow(average_face.reshape((100, 100)))
        # plt.show()

        # mean normalization
        self.norm_x = stacked_vector - self.mean_face
        return stacked_vector

    def display_data(self, data, title):
        nrows = 12
        ncols = 12
        figure, axes = plt.subplots(nrows, ncols)
        k = 0
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j].imshow(data[:, k].reshape((100, 100)))
                k += 1
        plt.title(title, loc='left')
        plt.show()

    def covariance(self):
        covariance_mat = np.dot(self.norm_x.T, self.norm_x) / self.m - 1
        return covariance_mat

    def eigen_value(self):
        eig_value, eig_vec = np.linalg.eig(self.covariance())
        # Convert lower dimension eigen vector to original dimension ui= A*vi(eigen vector relation between A.T*A and A*A.T)
        eig_face = np.dot(self.norm_x, eig_vec)
        # Normalize eigenvectors
        eig_face = eig_face / np.linalg.norm(eig_face, axis=0)
        # self.display_data(eig_face, 'Eigenface')
        return eig_value, eig_face

    def weights_calculation(self):
        _, eig_face = self.eigen_value()
        eig_face = eig_face[:, :25]
        weights = np.dot(self.norm_x.T, eig_face)
        reconstruction = self.mean_face+np.dot(eig_face, weights.T)
        self.display_data(reconstruction, 'reconstructed face')
        return weights, reconstruction

    def test(self):
        _, eig_face = self.eigen_value()
        eig_face = eig_face[:, :25]
        test_image = img.imread('./1.jpg')
        test_image = test_image.reshape(-1, 1)
        test_image = test_image - self.mean_face
        w = np.dot(test_image.T, eig_face)
        recon = self.mean_face + np.dot(eig_face, w.T)
        # plt.imshow(recon.reshape((100, 100)))
        return w,recon





a = Eigen('./cropped_image')
stacked=a.image_processing()
eigen_value,eigen_vectors = a.eigen_value()
weights,recons= a.weights_calculation()
test_w,recon = a.test()
