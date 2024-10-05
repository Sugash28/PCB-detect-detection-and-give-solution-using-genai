import tensorflow as tf
import numpy as np
import cv2
import json

# Load the model
model = tf.keras.models.load_model('keras_model.h5')

# Load labels
with open('label.txt', 'r') as f:
    labels = json.load(f)

# Function to preprocess the input image
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the image to the size your model expects (e.g., 224x224)
    image = cv2.resize(image, (224, 224))
    
    # Normalize the image
    image = image / 255.0  # Scale pixel values to [0, 1]
    
    # Expand dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    
    return image

# Function to predict using the model
def predict_pcb(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    
    # Get the predicted label index
    predicted_index = np.argmax(predictions[0])
    
    # Get the corresponding label
    predicted_label = labels[str(predicted_index)]
    
    return predicted_label

# Example usage
image_path = 'path/to/your/test_image.jpg'
predicted_label = predict_pcb(image_path)
print(f'Predicted PCB Label: {predicted_label}')
