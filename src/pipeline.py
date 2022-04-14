import os

import pandas as pd
from skimage import color
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from time import time

IMAGES_FOLDER_PATH = "../resized/images/"
ARTIST_CSV_PATH = "../artists.csv"
images = []
labels = []


def get_images_and_labels_from_path(path):
    """this function gets images and labels from a path"""
    for folder in os.listdir(path):
        for image_name in os.listdir(path + folder):
            # print(path + folder + "/" + image_name)
            image = cv2.imread(path + folder + "/" + image_name)
            image = cv2.resize(image, (100, 100))
            images.append(image)
            labels.append(folder)
        print("Images loaded from: " + folder)
    return images, labels


def flatten_images(images):
    flattened_images = []
    for image in images:
        flattened_images.append(image.flatten())
    return flattened_images


def get_colour_histograms(images):
    """this function gets colour histograms from images"""
    colour_histograms = []
    for image in images:
        colour_histograms.append(cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])[0])
    return colour_histograms


def read_csv(path):
    artists = pd.read_csv(path)
    artists.drop('id', axis=1, inplace=True)
    artists.drop('years', axis=1, inplace=True)
    artists.drop('bio', axis=1, inplace=True)
    artists.drop('wikipedia', axis=1, inplace=True)
    artists = artists.sort_values(by=['paintings'], ascending=False)
    artists.reset_index(inplace=True)
    return artists


def hog_features(img):
    features = []
    for image in img:
        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        features.append(fd)
    return features


def train_model(x, y):
    """this function trains a SVN model on data x and labels y in
    a 5 fold stratified manner"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = SVC(kernel='linear', C=1.0, probability=True, verbose=True)
    t0 = time()
    model.fit(x_train, y_train)
    print("training time:", round(time() - t0, 3), "s")
    return model, x_test, y_test


def test_model(model, x_test, y_test):
    """this function tests the model on data x and labels y"""
    y_pred = model.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    return y_pred


if __name__ == "__main__":
    get_images_and_labels_from_path(IMAGES_FOLDER_PATH)
    cv2.imshow("Example Resized image", images[0])
    print("Number of images: ", len(images))
    print("Number of labels: ", len(labels))
    waitKey = cv2.waitKey(0)
    hog = hog_features(images)
    colourHistograms = get_colour_histograms(images)
    rawPixels = flatten_images(images)
    # artistModel = train_model(rawPixels, labels)
    # test_model(artistModel[0], artistModel[1], artistModel[2])
    # print(labels)
    # print(read_csv(ARTIST_CSV_PATH))
    # for row in read_csv(ARTIST_CSV_PATH):
    #     print(row)
    # print(read_csv(ARTIST_CSV_PATH)["genre"])
