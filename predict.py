import numpy as np
from keras.models import load_model
from PIL import Image
model = load_model('mnist_cnn.h5')
def predict_digit(img):
    img = img.resize((28, 28)).convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255
    prediction = model.predict(img)
    return np.argmax(prediction)