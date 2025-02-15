from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
app = Flask(__name__)

dic = {0: 'Covid-19', 1: 'Normal', 2: 'Pneumonia'}

model = load_model('chest3.h5')

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(320, 320))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 320, 320, 3)
    label = model.predict(i)
    rounded_predictions = np.argmax(label, axis=1)
    print(rounded_predictions)
    if rounded_predictions == 0:
        final = 'COVID19'
    elif rounded_predictions == 1:
        final = 'NORMAL'
    elif rounded_predictions == 2:
        final = 'PNEUMONIA'
    # predictions = final
    percent = (label[0][rounded_predictions] * 100)
    predictions = '%s (%.2f%%)' % (final, percent)
    return predictions


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index2.html")



@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("index2.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True)
