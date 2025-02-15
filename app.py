from flask import Flask, render_template, request

import keras as keras
import cv2
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.applications.densenet import decode_predictions

import numpy as np

app = Flask(__name__)

model = load_model(r'C:\Users\sam\Desktop\flask\chest3.h5')

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index2.html')

@app.route('/submit', methods=['POST'])
def predict():
    imagefile = request.files['Imagefile']
    image_path = './images/' + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, (320,320))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    if image.shape[2] == 1:
        imgs = cv2.imread(image_path)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    else:
        imgs = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    imgs = cv2.resize(imgs,(320,320))
    imgs = imgs.reshape(1, 320, 320, 3).astype('float32')
    imgs = np.array(imgs) / 255.
    print(imgs.shape)
    label = model.predict(imgs)
    rounded_predictions = np.argmax(label, axis=1)
    print(rounded_predictions)
    if rounded_predictions == 0:
        final = 'COVID19'
    elif rounded_predictions == 1:
        final = 'NORMAL'
    elif rounded_predictions == 2:
        final = 'PNEUMONIA'
    #predictions = final
    percent = (label[0][rounded_predictions] * 100)
    predictions = '%s (%.2f%%)' % (final, percent)
    return render_template('index.html', prediction = predictions, img_path = image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)