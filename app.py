import sys
from flask import Flask, jsonify, request, make_response
import numpy as np
from PIL import Image
import base64
import re
from io import StringIO, BytesIO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image# Helper libraries
import json

# load model
modelFileName = 'last_model.h5'
model = keras.models.load_model('./models/{}'.format(modelFileName))

# app
app = Flask(__name__)

def maskImage(base64_img):
    # base64_img = file = request.files['canvas'].read() 
    base64_img = re.sub('^data:image/.+;base64,', '', base64_img)
    base64_img_bytes = base64_img.encode('utf-8')
    with open('decoded_image.png', 'wb') as file_to_save:
        decoded_image_data = base64.decodebytes(base64_img_bytes)
        file_to_save.write(decoded_image_data)
    # decoded_image_data = base64.decodebytes(base64_img_bytes)
    return 'decoded_image.png'

# routes
@app.route('/', methods=['GET'])
def base():
    out = 'hello world'
    return out

@app.route('/run', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)['imageBase64']
    fileName = maskImage(data)
    img = image.load_img('./decoded_image.png', target_size=(128, 128), grayscale=True)
    # npimg = np.array(img)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    # img_preprocessed = preprocess_input(img_batch)
    result = model.predict(img_batch)
    
    with open('categories.txt') as f:
        labels = [i for i in f.readlines()]
    # probs = np.argmax(result,axis=1)
    li = result[0].tolist()
    print(li)
    result = ""
    probs = li.index(max(li))
    li[probs] = 0
    probs = li.index(max(li))
    li[probs] = 0
    probs = li.index(max(li))
    print(li[250])
    print(li[120])
    print(max(li))
    print(probs)
    output = {'output': labels[probs]}
    # return data
    out = json.dumps(output)
    return make_response(out)

@app.after_request
def after_request(response):
    # print("log: setting cors", file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == '__main__':
    app.run(port = 5000, debug=True)