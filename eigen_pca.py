import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os


def image_processing():
    m = len(os.listdir('./cropped_image'))
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

    average_face = np.sum(stacked_vector, axis=1)/m
    # Display Average face
    plt.imshow(average_face.reshape((100, 100)))
    # plt.title("average face")
    # check is the operations till now are properly done
    # plt.imshow(stacked_vector[:, 0].reshape((100, 100)))

    # mean normalization
    normalized_vec = stacked_vector - average_face.reshape(10000,1)
    return normalized_vec, average_face


norm_x, mean_face = image_processing()


def covariance():
    m = norm_x.shape[1]
    covariance_mat = np.dot(norm_x.T, norm_x) / m - 1
    return covariance_mat


def eigen_value():
    eig_value, eig_vec = np.linalg.eig(covariance())
    # Convert lower dimension eigen vector to original dimension ui= A*vi(eigen vector relation between A.T*A and A*A.T)
    eig_face = np.dot(norm_x, eig_vec)
    return eig_value, eig_face


eig_value, eig_face = eigen_value()


def display_eigenfaces():
    nrows = 10
    ncols = 10
    figure, axes = plt.subplots(nrows, ncols)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].imshow(eig_face[:, k].reshape((100, 100)))
            k += 1
    plt.show()


display_eigenfaces()

# wi = mean_face + np.dot(norm_x.T, eig_face)
# plt.imshow(wi[:, 7].reshape((100, 100)))

