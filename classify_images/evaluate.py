import cv2
import os
import numpy as np

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR) # this goes *before* tf import

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential



dsize = (400, 400)


def load_iamge(path):
    img = cv2.imread(os.path.join(path))
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    output = cv2.resize(gray_image, dsize)
    return output

def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

layer_1 = Dense(units=250, activation="relu")
layer_2 = Dense(units=150, activation="relu")
layer_3 = Dense(units=15, activation="relu")
layer_4 = Dense(units=1, activation="sigmoid")

model = Sequential([tf.keras.Input(shape=(160000,)), layer_1, layer_2, layer_4])
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

checkpoint_dir = '/home/tingxfan/machine_learning/class_images/training_5/'

latest = tf.train.latest_checkpoint(checkpoint_dir)
print('-'*20)
print(latest)

# Load the previously saved weights
model.load_weights(latest)

model.summary()

# test_path = '/home/tingxfan/machine_learning/class_images/test_images/peter-burdon-nvcxiFjWki0-unsplash.jpg' # 0
test_path = '/home/tingxfan/machine_learning/class_images/test_images/joshua-fuller-UJ7t9uNcTDU-unsplash.jpg' # 1
# test_path = '/home/tingxfan/machine_learning/class_images/test_images/jeremy-perkins-UgNjyPkphtU-unsplash.jpg' # 0
test_img = load_iamge(test_path)
X_test = np.reshape(test_img, (1,-1))

show_image(test_img)

predictions = model.predict(X_test)
print("predictions = " + str(predictions))


