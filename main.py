from face_detect import FaceDetect
import train_test_split
from eigen_pca import Eigen
from test import Test
import os
from constants import *


def main():
    if not os.path.isdir(training_set):
        print("Cropping faces and making train and test sets")
        face = FaceDetect()
        face.face_return()
        train_test_split.split()

    print("Training....")
    train_set = Eigen(training_set)
    train_image_label = train_set.label_extract()
    train_stacked_images, mean_face = train_set.image_processing()
    _, eig_face = train_set.eigen_value()

    # selecting no of eigen faces, the first values are the largest ones no need to sort them
    eig_face = eig_face[:, :no_of_eigfaces]
    train_weights, recons_train = train_set.weights_calculation(eig_face, mean_face)

    # display selected eigen faces
    # train_set.display_data(eig_face, 'Selected Eigen faces')
    #
    # # display original images
    # train_set.display_data(train_stacked_images, 'Original faces')
    #
    # # display reconstructed training face
    # train_set.display_data(recons_train, 'reconstructed training data')

    print("Training finished, testing ......")

    test_set = Eigen(testing_set)
    test_image_label = test_set.label_extract()
    print('Original label:', test_image_label)
    test_stacked_images, _ = test_set.image_processing(mean_face)
    test_weights, recons_test = test_set.weights_calculation(eig_face, mean_face)

    # display original test face
    test_set.display_data(test_stacked_images, 'original testfaces')

    # display reconstructed test face
    test_set.display_data(recons_test, 'reconstructed test data')
    test = Test(train_stacked_images, train_weights, test_weights)
    predicted_label = test.match_index(train_image_label)
    print('Predicted label:', predicted_label)
    match_check = [1 if tl == pl else 0 for tl, pl in zip(test_image_label, predicted_label)]
    print(match_check)
    print("Accuracy:", sum(match_check)/len(predicted_label))

if __name__ == "__main__":
    main()
