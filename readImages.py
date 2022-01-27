import os
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

print('hello world!')
IMAGES_FOLDER_PATH = "./images/images/"
images = []
labels = []


def get_label_name(file_name):
    label_name = ' '
    artist = file_name.split('_')
    artist.pop()
    label_name = label_name.join(artist)
    print(label_name)
    return label_name


# for folder in os.listdir(IMAGES_FOLDER_PATH):
#     print(folder)
#     for image in os.listdir("./images/images/" + folder):
#         file_path = IMAGES_FOLDER_PATH + folder + "/" + str(image)
#         labels.append(get_label_name(image))
#         images.append(plt.imread(file_path))

foldersTest = ['Andrei_Rublev', 'Salvador_Dali', 'Henri_Matisse']
for folder in foldersTest:
    for image in os.listdir("./images/images/" + folder):
        file_path = IMAGES_FOLDER_PATH + folder + "/" + str(image)
        labels.append(get_label_name(image))
        images.append(plt.imread(file_path))


figure, axes = plt.subplots(nrows=5, ncols=5)
axes = axes.flatten()
for ax in range(len(axes)):
    axes[ax].set_axis_off()
    axes[ax].imshow(color.rgb2gray(images[ax]), cmap=plt.cm.gray_r)
    axes[ax].set_title("Training: %s" % labels[ax])

# flatten the images
n_samples = len(images)
processed = []
features = []
for image in images:
    imageTemp = image
    imageGray = color.rgb2gray(imageTemp)
    resized = np.resize(imageGray, (100, 100)).reshape(-1)
    processed.append(resized)
processed = np.array(processed)

# Split data into 75% train and 25% test subsets
training_images, testing_images, training_labels, testing_labels = train_test_split(
    processed, labels, test_size=0.25, random_state=2, shuffle=True
)

# Create a classifier: KNN
clf = SVC()
# Train classifier
clf.fit(training_images, training_labels)

# Test classifier
predicted = clf.predict(testing_images)

# How accurate was the classifier?
total = 0
correct = 0
accuracy = 0.0
for prediction, actual in zip(predicted, testing_labels):
    print("Predicted: " + str(prediction) + " Actual: " + str(actual))
    if actual == prediction:
        correct += 1
    total += 1
accuracy = correct / total
print(accuracy)
acc = clf.score(testing_images, testing_labels)
print(acc)

_, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 3))
axes = axes.flatten()
for ax, image, prediction in zip(axes, training_images, predicted):
    ax.set_axis_off()
    image = image.reshape(100, 100)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(testing_labels, predicted)}\n"
)

plt.show()

print()
