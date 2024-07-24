from keras.models import load_model
from PIL import Image
from flask import Flask, request, render_template
import io
import numpy as np

app = Flask(__name__)
model = load_model('mnist_cnn.h5')

def predict_digit(img):
    img = img.resize((28, 28)).convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255
    prediction = model.predict(img)
    return np.argmax(prediction)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read()))
        digit = predict_digit(img)
        return render_template("index.html", digit=digit)
    return render_template("index.html", digit=None)

if __name__ == "__main__":
    app.run(debug=True)
