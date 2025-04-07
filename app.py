import os
import cv2
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request
from keras_preprocessing import image
from googlesearch import search
from io import BytesIO
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__, template_folder='templates')

# Load solutions from dataset
solutions_df = pd.read_excel('leaf_disease_dataset.xlsx')
solutions_df['Class'] = solutions_df['Class'].str.replace('-', '_')
solutions = dict(zip(solutions_df['Class'], solutions_df['Solution']))

# Load pre-trained models for ensemble
inception_model = tf.keras.models.load_model('models/inceptionV3.h5')
resnet_model = tf.keras.models.load_model('models/Resnet50.h5')
vgg_model = tf.keras.models.load_model('models/VGG16.h5')

inception_model.name = "InceptionV3"
resnet_model.name = "ResNet50"
vgg_model.name = "VGG16"

models = [inception_model, resnet_model, vgg_model]

# List of class labels
labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
] 

def google_search(query, num_results=5):
    try:
        return list(search(query, num_results=num_results))
    except Exception as e:
        print(f"Google search failed: {str(e)}")
        return []

def preprocess_image(image_array):
    img = cv2.resize(image_array, (128, 128))
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def frontpage():
    if request.method == 'POST':
        image_array = None
        image_filename = None

        # Handle webcam capture (base64 string)
        if 'webcam_image' in request.form and request.form['webcam_image']:
            webcam_data = request.form['webcam_image'].split(',')[1]
            img_bytes = base64.b64decode(webcam_data)
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
            image_array = np.array(img)
            image_filename = "captured_from_webcam.jpg"
            image_path = os.path.join('static/uploads', image_filename)
            img.save(image_path)

        # Handle file upload
        elif 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                image_filename = uploaded_file.filename
                if not os.path.exists('static/uploads'):
                    os.makedirs('static/uploads')
                image_path = os.path.join('static/uploads', image_filename)
                uploaded_file.save(image_path)
                img = image.load_img(image_path, target_size=(128, 128))
                image_array = image.img_to_array(img)

        if image_array is not None:
            img_array = preprocess_image(image_array)

            votes = np.zeros(len(labels))
            model_votes = {model.name: [] for model in models}
            for model in models:
                predictions = model.predict(img_array)
                predicted_index = np.argmax(predictions)
                votes[predicted_index] += 1
                model_votes[model.name].append(labels[predicted_index])

            predicted_class_index = np.argmax(votes)
            predicted_class_label = labels[predicted_class_index]
            solution = solutions.get(predicted_class_label, 'No specific solution available.')

            query = f"{predicted_class_label} disease solutions"
            search_results = google_search(query)

            context = {
                "predicted_class": predicted_class_label,
                "solution": solution,
                "image_path": image_path,
                "image_filename": image_filename,
                "search_results_json": search_results,
                "model_votes": model_votes
            }
            print(context)

            return render_template('result.html', **context)

    return render_template('frontpage.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')               
def about():                      
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=False)
  