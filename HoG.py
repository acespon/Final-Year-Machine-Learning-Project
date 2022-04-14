import numpy as np
import matplotlib.pyplot as plt
import os
import math


def extract_HOG(image):
    feature_vector = []

    return feature_vector



def get_label_name(file_name):
    label_name = ' '
    artist = file_name.split('_')
    artist.pop()
    label_name = label_name.join(artist)
    # print(label_name)
    return label_name


IMAGES_FOLDER_PATH = "./images/images/"
labels = []
images = []
if __name__ == "__main__":

    foldersTest = ['Salvador_Dali']
    for folder in foldersTest:
        for image in os.listdir("./images/images/" + folder)[0:1]:
            file_path = IMAGES_FOLDER_PATH + folder + "/" + str(image)
            labels.append(get_label_name(image))
            images.append(plt.imread(file_path))

    img = images[0]
    gray = np.mean(img, axis=2)
    print(gray.dtype)
    print(np.shape(img))
    print(np.shape(gray))

    rows, cols = np.shape(gray)
    Ix = gray
    Iy = gray

    for i in range(rows - 2):
        Iy[i, :] = (gray[i, :] - gray[i + 2, :])

    for i in range(cols - 2):
        Ix[:, i] = (gray[:, i] - gray[:, i + 2])

    angle = np.arctan(Ix / Iy)
    print(angle)
    angle = angle + 90
    print(angle)
    magnitude = np.sqrt(np.square(Ix) +np.square(Iy))


    print("%s wide and %s tall" % (cols, rows))

    imgplot = plt.imshow(Iy)

    plt.set_cmap('gray')
    plt.show()

    print("Hello World!")
