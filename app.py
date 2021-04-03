import pandas as pd
from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
import base64
import re
from io import StringIO, BytesIO
from tensorflow import keras


# load model
modelFileName = 'quickdraw_init_model.h5'
model = keras.models.load_model('./models/{}'.format(modelFileName))

# app
app = Flask(__name__)

def maskImage(base64_img):
    # base64_img = file = request.files['canvas'].read() 
    base64_img = re.sub('^data:image/.+;base64,', '', base64_img)
    base64_img_bytes = base64_img.encode('utf-8')
    decoded_image_data = base64.decodebytes(base64_img_bytes)
    return decoded_image_data

# routes
@app.route('/', methods=['GET'])
def base():
    print('hello world')

@app.route('/run', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)['imageBase64']
    maskImage(data)

    result = model.predict(data)
    # send back to browser
    output = {'results': int(result[0])}
    # return data
    return jsonify(results=output)

@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == '__main__':
    app.run(port = 5000, debug=True)