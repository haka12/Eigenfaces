from face_detect import FaceDetect
import train_test_split
from eigen_pca import Eigen
from test import Test


def main():
    face = FaceDetect()
    face.face_return()
    train_test_split.split()
    # eig = Eigen('./cropped_image')
    # eig.image_processing()
    # weights = eig.weights_calculation()


if __name__ == "__main__":
    main()
