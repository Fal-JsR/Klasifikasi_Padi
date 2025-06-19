import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops

def process_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))

    # Ekstraksi fitur GLCM
    glcm = graycomatrix(resized, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Ekstraksi fitur bentuk (Eccentricity)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labeled = label(thresh)
    props = regionprops(labeled)

    eccentricity = props[0].eccentricity if props else 0.0

    # Gabung fitur ke model
    features = [[eccentricity, contrast, correlation, energy, homogeneity]]
    model = joblib.load('model/model_svm.pkl')
    prediction = model.predict(features)[0]

    return prediction, {
        'Eccentricity': eccentricity,
        'Contrast': contrast,
        'Correlation': correlation,
        'Energy': energy,
        'Homogeneity': homogeneity
    }

def save_edge_image(image_path, output_path):
    img = cv2.imread(image_path, 0)
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite(output_path, edges)