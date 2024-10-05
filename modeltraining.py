from flask import Flask, request, render_template, send_file
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Set paths
output_dir = 'G:/intel/PCB_DATASET/output'
best_model_path = os.path.join(output_dir, 'yolov8n.pt')  # Adjust the path to the best model
model = YOLO(best_model_path)

@app.route('/')
def home():
    return render_template('index.html')  # A simple form to upload images

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Load the image
    img = Image.open(file.stream)

    # Save the image temporarily
    temp_image_path = os.path.join(output_dir, 'temp_image.jpg')
    img.save(temp_image_path)

    # Run inference on the image
    results = model(temp_image_path, imgsz=640, conf=0.25, save=True, save_txt=True, save_conf=True)

    # Display the image with bounding boxes
    results_img = results[0].plot()  # This will plot the image with predictions (bounding boxes)

    # Convert the image to byte array for displaying in Flask
    img_byte_arr = io.BytesIO()
    results_img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/jpeg')

@app.route('/evaluate', methods=['GET'])
def evaluate():
    # Run evaluation on the validation dataset
    metrics = model.val()  # This evaluates the model on the validation dataset
    return metrics  # You can return this as JSON or render in a template

if __name__ == '__main__':
    app.run(debug=True)
