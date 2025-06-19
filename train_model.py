import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv('fitur_dataset.csv')
X = df[['Eccentricity', 'Contrast', 'Correlation', 'Energy', 'Homogeneity']]
y = df['Label']

# Cek distribusi label
print("\n📊 Distribusi Label:\n", y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model SVM
model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("\n📊 Akurasi:", accuracy_score(y_test, y_pred))
print("\n📋 Laporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Simpan model
joblib.dump(model, 'model/model_svm.pkl')
print("✅ Model disimpan sebagai model/model_svm.pkl")
