import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from tqdm import tqdm

def extract_features(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))

    glcm = graycomatrix(resized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labeled = label(thresh)
    props = regionprops(labeled)
    eccentricity = props[0].eccentricity if props else 0.0

    return [eccentricity, contrast, correlation, energy, homogeneity]


dataset_path = 'dataset'
labels = ['matang', 'belum_matang']
data = []

for class_label in labels:
    folder = os.path.join(dataset_path, class_label)
    for filename in tqdm(os.listdir(folder), desc=f'Memproses {class_label}'):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            fitur = extract_features(path)
            fitur.append(class_label)
            data.append(fitur)

df = pd.DataFrame(data, columns=['Eccentricity', 'Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Label'])
df.to_csv('fitur_dataset.csv', index=False)
print('✅ Fitur berhasil diekstrak dan disimpan ke fitur_dataset.csv')