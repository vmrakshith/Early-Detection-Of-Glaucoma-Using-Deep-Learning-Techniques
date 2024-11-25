import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Define the Flask app
app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'C:/Users/KIRAN KUMAR/Glaucoma-Detection-using-CNN/split/saved_model/my_model4.h5'
model = load_model(MODEL_PATH)

# Define a function to preprocess the uploaded image for prediction
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Change size based on your model input
    img_array = img_to_array(img) / 255.0  # Scale the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define the home route
@app.route('/')
def index():
    return render_template('index.html')

# Define the route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the file temporarily
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Preprocess the image
        processed_image = preprocess_image(image_path)

        # Make a prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)  # Assuming you have a softmax output

        # Clean up the temporary file
        os.remove(image_path)

        # Return the result (this can be customized based on your model’s output)
        result = "Glaucoma detected" if predicted_class[0] == 1 else "No Glaucoma detected"
        
        return render_template('result.html', prediction=result)

# Run the app
if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)
