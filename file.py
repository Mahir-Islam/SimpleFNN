'''
1. Make sure you have Tensorflow installed.
If not, then run pip install tensorflow in CMD
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

'''
2. These values determine how much data you would like to train on.
Using batches makes the training more efficient
'''
BATCH_SIZE = 200
EPOCHS = 10

'''
3. These commands grab the data from the MNIST datasets, which consists of
thousands of 28x28 images of handwritten number digits.
60,000 images will be used for training, and 10,000 for testing.
The y-values are then 1-hot encoded to turn them into a categorical variable,
since recognising a digit is nominal (despite having a numerical outcome).
'''
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1).reshape(60000,28,28,1)
x_test  = keras.utils.normalize(x_test , axis=1).reshape(10000,28,28,1)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test  = keras.utils.to_categorical(y_test, num_classes=10)

'''
4. This is the main model. The input layer takes a 28x28 image as input.
The next layers have 128 nodes each, with each node having a bias, and each
connection having a weight.

The output layer is activated with softmax, which normalises the output to
have a sum of one, which makes it a valid probability distribution.
'''

model = keras.Sequential([
    layers.Flatten(input_shape=(28,28)),

    layers.Dense(128, activation='relu'),
    #layers.Dropout(rate=0.1),
    #layers.BatchNormalization(),

    layers.Dense(128, activation='relu'),
    #layers.Dropout(rate=0.1),
    #layers.BatchNormalization(),

    layers.Dense(10, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
)

model.fit(
    x_train,
    y_train,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS
    )

model.save('MNIST-BASIC')