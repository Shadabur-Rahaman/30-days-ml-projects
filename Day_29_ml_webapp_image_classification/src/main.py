from flask import Flask, render_template, request
from src.predict import predict_image_class
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded."

    image_file = request.files['image']
    if image_file.filename == '':
        return "Empty filename."

    image_path = os.path.join('static', image_file.filename)
    image_file.save(image_path)

    prediction = predict_image_class(image_path)
    return render_template('result.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
