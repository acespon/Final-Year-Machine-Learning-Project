# need those 2 to import and init
import os
import pickle
import cv2
import typer
import pipeline

app = typer.Typer()


# commands go here
# sample command
@app.command()
def main(path: str = "../resized/images/", maxsamples: int = 150, minsamples: int = 50, width: int = 100,
         height: int = 100):
    feature_extraction = -1
    while feature_extraction not in ['1', '2', '3', '4', '5']:
        feature_extraction = typer.prompt("Select feature extraction method: \n1. Raw Pixels\n2. HoG\n3. Colour "
                                          "Histogram\n4. HoG and Colour Histograms\n5. Raw Pixels, HoG, "
                                          "Colour Histograms\n")

    images, labels = pipeline.get_images_and_labels_from_path(path, maxsamples, minsamples, width, height)
    samples = []

    print("Total number of images: " + str(len(images)))

    if feature_extraction == '1':
        samples = pipeline.flatten_images(images)
    elif feature_extraction == '2':
        samples = pipeline.hog_features(images)
    elif feature_extraction == '3':
        samples = pipeline.calculate_color_histogram(images)
    elif feature_extraction == '4':
        samples = pipeline.HoG_ColHist(images)
    elif feature_extraction == '5':
        samples = pipeline.Raw_HoG_ColHist(images)

    model = pipeline.train_model(samples, labels)

    model_name = typer.prompt("Enter model name: ")
    pipeline.save_model(model, "../models/%s.pkl" % model_name)

    pipeline.cmatrix(model)


@app.command()
def test(imagepath: str = "../testing/", modelpath: str = "../models/", imagewidth: int = 100,
         imageheight: int = 100):
    images = []
    labels = [os.path.splitext(filename)[0] for filename in os.listdir(imagepath)]
    for sample in os.listdir(imagepath):
        image = cv2.imread(imagepath + sample)
        image = cv2.resize(image, (imagewidth, imageheight))
        images.append(image)

    feature_extraction = -1
    while feature_extraction not in ['1', '2', '3', '4', '5']:
        feature_extraction = typer.prompt("Select feature extraction method: \n1. Raw Pixels\n2. HoG\n3. Colour "
                                          "Histogram\n4. HoG and Colour Histograms\n5. Raw Pixels, HoG, "
                                          "Colour Histograms\n")

    test_image = []
    if feature_extraction == '1':
        test_image = pipeline.flatten_images(images)
    elif feature_extraction == '2':
        test_image = pipeline.hog_features(images)
    elif feature_extraction == '3':
        test_image = pipeline.calculate_color_histogram(images)
    elif feature_extraction == '4':
        test_image = pipeline.HoG_ColHist(images)
    elif feature_extraction == '5':
        test_image = pipeline.Raw_HoG_ColHist(images)

    model = ""
    while model not in os.listdir(modelpath):
        model = typer.prompt("Enter model name: ")
    model = pickle.load(open(modelpath + model, 'rb'))
    print(model)
    y_pred = model.predict(test_image)
    print(y_pred)
    print(pipeline.metrics.classification_report(labels, y_pred))


# need this to run at the bottom
if __name__ == "__main__":
    app()
