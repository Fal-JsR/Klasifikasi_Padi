from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from model.utils import process_image, save_edge_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['EDGE_FOLDER'] = 'static/edges'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "❌ Tidak ada file yang diunggah"

    file = request.files['image']
    if file.filename == '':
        return "❌ File belum dipilih"

    if file:
        # Simpan gambar asli
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Proses klasifikasi gambar
        raw_result, glcm_features = process_image(filepath)

        # Ubah hasil prediksi menjadi label yang deskriptif
        if raw_result == 'matang':
            result = "🌾 Padi Matang Siap Panen"
        elif raw_result == 'belum_matang':
            result = "🌱 Padi Belum Matang"
        else:
            result = f"❓ Tidak Dikenali: {raw_result}"

        # Simpan hasil deteksi tepi
        edge_filename = 'edge_' + filename
        edge_path = os.path.join(app.config['EDGE_FOLDER'], edge_filename)
        save_edge_image(filepath, edge_path)

        # Tampilkan hasil ke result.html
        return render_template('result.html',
                               result=result,
                               filename=filename,
                               edge_filename=edge_filename,
                               glcm=glcm_features)

if __name__ == '__main__':
    app.run(debug=True)
