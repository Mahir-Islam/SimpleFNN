import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#Make sure you also have Matplotlib, CV2 and Numpy installed

MODEL_NAME = 'MNIST-BASIC'
model = keras.models.load_model(MODEL_NAME)

'''
Rename this to any png in the testimages file.
Make sure it has a black background, with white text and is 28x28 pixels.
If it is white background and black text, uncomment line 20.
'''
image_name = "five"

img = cv2.imread(f'testimages/{image_name}.png')[:,:,0]
#img = np.invert(img)
img = np.reshape(img,(1,28,28))

prediction = model.predict(img)
print(f'Predicting {image_name}.png as {np.argmax(prediction)}.\nRaw Output: {np.around(prediction, decimals=2)}')
