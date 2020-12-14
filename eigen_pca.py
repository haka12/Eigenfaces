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
    average_face = np.zeros((prev_column.size, 1))

    # stacking the column vectors to one vector N^2 * m
    for count, _ in enumerate(column_vectors):
        if count + 1 == len(column_vectors):
            break
        average_face = average_face + column_vectors[count]
        stacked_vector = np.column_stack((prev_column, column_vectors[count + 1]))
        prev_column = stacked_vector

    average_face = average_face / m

    # Display Average face
    plt.imshow(average_face.reshape((100, 100)))
    plt.title("average face")
    # plt.imshow(stacked_vector[:, 0].reshape((100, 100)))

    # mean normalization
    normalized_vec = stacked_vector - average_face
    return normalized_vec


def covariance():
    norm_x = image_processing()
    m = norm_x.shape[1]
    covariance_mat = np.dot(norm_x, norm_x.T) / m
    return covariance_mat

