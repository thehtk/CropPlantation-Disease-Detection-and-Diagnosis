from __future__ import division, print_function


import numpy as np

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


MODEL_PATH = "C:\\Users\\Hp\\OneDrive\\Sem7\\Project\\model2.h5"

# Load  trained model
model = load_model(MODEL_PATH)
model.make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(image_path, model):
    img = image.load_img(image_path, target_size=(200, 200))

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

    
    class_labels =['Apple Apple scab','Apple_Black_rot', 'Apple Cedar_apple_rust', 'Applle...healthy', 'Blueberry...healthy', 
                   'Cherry (including sour)...Powdery mildew','Cherry (including sour)...healthy', 
                   'Corn (maize).Cercospora leaf spot Gray.1 eaf spot', 'Corn (maize)...Common rust_', 'Corn (maize)...Northern_Leaf Blight', 
                   'Corn (maize)...healthy', 'Grape_Black_rot', 'Grape__Esca(Black_Measle s)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
                   'Grape...healthy', 'Orange Haunglongbing (Citrus greening)', 'Peach Bacterial spot', 'Peach_hea lthy', 'Pepper', 
                   'bell_Bacterial spot', 'Pepper, bell...healthy', 'Potato Early blight', 'Potato._Late_blight', 'Potato...healthy', 
                   'Raspberry... healthy', 'Soybeanhealthy', 'Squash_Powdery mildew', 'Strawberry_Le af scorch', 'Strawberry...healthy', 
                   'Tomato Bacterial spot', 'Tomato Early blight','Tomato Late blight', 'Tomato Leaf_Mold', 'Tomato Sept oria leaf spot',
                   'Tomato Spider mites Two-spotted spider mite''Tomato.Target Spot','Tomato Tomato Yellow Leaf Curl_Virus', 
                   'Tomato mosaic virus','Tomato healthy']
    class_labels =['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    
    predicted_class_label = class_labels[predicted_class_index]


    print(f"Predicted class: {predicted_class_label}")
    print(f"Class probabilities: {preds[0]}")
    li=[predicted_class_label]

  ########################################################################################

    # class_labels=[f'Class{i}' for i in range(3,34)]
    # predicted_class_label=class_labels[predicted_class_index]
  
    return render_template('index.html',prediction=li)



if __name__ == '__main__':
    app.run(debug=True)
