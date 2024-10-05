import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, url_for
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io
import glob
import base64

app = Flask(__name__)

# Set paths
OUTPUT_DIR = 'G:/intel/PCB_DATASET/output'  # Adjust the path as needed
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'yolov8n.pt')
CLASSES = ['Missing Hole', 'Mouse Bite', 'Open Circuit', 'Short', 'Spurious Copper']

# Load the YOLOv8 model
model = YOLO(BEST_MODEL_PATH)

def yolo_to_original_annot(image_name, yolo_labels, image_shape):
    original_annot = []
    original_width, original_height = image_shape[1], image_shape[0]

    for yolo_label in yolo_labels:
        class_index, x_center, y_center, bbox_width, bbox_height, confidence = yolo_label

        original_x_center = x_center * original_width
        original_y_center = y_center * original_height
        original_bbox_width = bbox_width * original_width
        original_bbox_height = bbox_height * original_height
        
        original_x_min = original_x_center - original_bbox_width / 2
        original_y_min = original_y_center - original_bbox_height / 2
        original_x_max = original_x_center + original_bbox_width / 2
        original_y_max = original_y_center + original_bbox_height / 2

        original_annot.append({
            'filename': image_name,
            'width': int(original_width),
            'height': int(original_height),
            'class': CLASSES[int(class_index)],
            'xmin': int(original_x_min),
            'ymin': int(original_y_min),
            'xmax': int(original_x_max),
            'ymax': int(original_y_max),
            'confidence': confidence
        })

    return pd.DataFrame(original_annot)

def visualize_annotations(image, original_annotations):
    img = image.copy()
    
    for _, row in original_annotations.iterrows():
        class_name = row['class']
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        label = f"{class_name}: {row['confidence']:.2f}"
        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_byte_arr = io.BytesIO()
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return img_byte_arr

def process_single_image(image_path):
    img = cv2.imread(image_path)
    image_name = os.path.basename(image_path)
    
    yolo_results = model(img, imgsz=640, conf=0.25)
    yolo_labels = yolo_results[0].boxes.data.tolist()

    original_annotations = yolo_to_original_annot(image_name, yolo_labels, img.shape)
    
    annotated_image = visualize_annotations(img, original_annotations)
    
    return annotated_image, original_annotations.to_dict('records')

def process_images(image_dir):
    results = []
    image_files = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))

    for image_path in image_files:
        image_name = os.path.basename(image_path)
        annotated_image, predictions = process_single_image(image_path)
        
        # Convert image to base64 for embedding in HTML
        img_base64 = base64.b64encode(annotated_image.getvalue()).decode('utf-8')

        results.append({
            'image_name': image_name,
            'predictions': predictions,
            'image_data': img_base64
        })

    return results

@app.route('/')
def index():
    return render_template('indexx.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    image_name = file.filename
    image_path = os.path.join(OUTPUT_DIR, image_name)
    file.save(image_path)

    annotated_image, predictions = process_single_image(image_path)
    
    # Convert image to base64 for embedding in HTML
    img_base64 = base64.b64encode(annotated_image.getvalue()).decode('utf-8')

    return render_template('result.html', image_name=image_name, 
                           img_data=img_base64, predictions=predictions)

@app.route('/process', methods=['POST'])
def process_pcb_images():
    image_dir = request.form.get('image_dir', OUTPUT_DIR)
    results = process_images(image_dir)
    return render_template('multiple_results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)