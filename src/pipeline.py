import os

import pandas as pd
from matplotlib import pyplot as plt
from skimage import color
import numpy as np
import cv2
from skimage.feature import hog, daisy
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from time import time
import pickle


IMAGES_FOLDER_PATH = "../resized/images/"
ARTIST_CSV_PATH = "../artists.csv"


def get_images_and_labels_from_path(path, max, min, imageWidth, imageHeight):
    """this function gets images and labels from a path"""
    images = []
    labels = []
    for folder in os.listdir(path):
        count = 0
        if len(os.listdir(path + folder)) < min:
            continue
        for image_name in os.listdir(path + folder):
            image = cv2.imread(path + folder + "/" + image_name)
            image = cv2.resize(image, (imageWidth, imageHeight))
            images.append(image)
            labels.append(folder)
            count += 1
            if count == max:
                break
        print("Images loaded from: " + folder)
    return images, labels


def generate_flipped_images(images):
    """this function generates flipped images"""
    imgs = images
    flipped_images = []
    for image in imgs:
        flipped_images.append(cv2.flip(image, 1))
    return imgs + flipped_images


def duplicate_labels(labels):
    """this function duplicates the labels"""
    for i in range(len(labels)):
        labels.append(labels[i])
    return labels


def flatten_images(images):
    flattened_images = []
    for image in images:
        flattened_images.append(image.flatten())
    return flattened_images


def calculate_color_histogram(images):
    """this function calculates the color histogram of an image"""
    histograms = []
    for image in images:
        histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histograms.append(histogram.flatten())
    histograms = normalize(histograms, norm='l2')
    return histograms


def HoG_ColHist(images):
    vectors = []
    for image in images:
        hogfeature = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        colour_histogram = calculate_color_histogram([image]).flatten()
        vectors.append(np.concatenate((hogfeature, colour_histogram)))
    return vectors


def Raw_HoG_ColHist(images):
    vectors = []
    for image in images:
        hogfeature = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        colour_histogram = calculate_color_histogram([image]).flatten()
        gray = color.rgb2gray(image)
        vectors.append(np.concatenate((hogfeature, colour_histogram, gray.flatten())))
    return vectors


def count_classes(labels):
    """this function counts the number of classes within the dataset"""
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def delete_class_from_dataset(images, labels, class_name):
    """this function deletes a class from the dataset"""
    for i in range(len(labels)):
        if labels[i] == class_name:
            del images[i]
            del labels[i]
    return images, labels


def trim_classes(images, labels, counts, min):
    """this function trims the dataset to only include classes with
    at least min paintings and at most max paintings"""
    imgs = images
    labs = labels
    for artist in counts:
        if counts[artist] < min:
            imgs, labs = delete_class_from_dataset(imgs, labs, artist)
    return imgs, labs


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
    count = 100
    for image in img:
        if count % 100 == 0:
            print("HOG features extracted: " + str(count))
        gray_image = color.rgb2gray(image)
        fd, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                            feature_vector=True)
        features.append(fd)
        count += 1
    return features


def train_model(x, y):
    """this function trains a SVN model on data x and labels y in
    a 5 fold stratified manner"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced')
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


def cmatrix(model):
    """this function returns the confusion matrix of the model"""
    cm = confusion_matrix(model[2], test_model(model[0], model[1], model[2]), labels=model[0].classes_)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model[0].classes_)
    disp.plot(cmap='Blues')
    plt.show()
    return disp.confusion_matrix


def save_model(model, filename):
    pickle.dump(model[0], open(filename, 'wb'))


def load_model(filename):
    filename = filename + ".sav"
    return pickle.load(open(filename, 'rb'))


if __name__ == "__main__":
    print("Main")
