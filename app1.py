from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


MODEL_PATH = 'models/trained_model.h5'

# Load  trained model
model = load_model(MODEL_PATH)
model.make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/',methods=['GET'])
def hello_word():
    return render_template('index.html')
@app.route('/',methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    preds = model_predict(image_path, model)

    predicted_class_index=np.argmax(preds)
    ######################################################################################
    print(predicted_class_index)

    class_labels = ['Apple__Apple_scab','Apple_Black_rot','Apple_Cedar_apple_rust','Apple_healthy','Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot','Corn(maize)__Common_rust','Corn_(maize)__healthy','Grape__Black_rot',
    'Grape__Esca(Black_Measles)','Grape__healthy','Potato_Early_blight','Potato_healthy','Potato_Late_blight','Tomato__Bacterial_spot',
    'Tomato__healthy','Tomato_Leaf_Mold','Tomato__Tomato_mosaic_virus']
    predicted_class_label = class_labels[predicted_class_index]


    print(f"Predicted class: {predicted_class_label}")
    print(f"Class probabilities: {preds[0]}")


  ########################################################################################

    # class_labels=[f'Class{i}' for i in range(3,34)]
    # predicted_class_label=class_labels[predicted_class_index]
  
    return render_template('index.html',prediction=predicted_class_label)



if __name__ == '__main__':
    app.run(debug=True)
