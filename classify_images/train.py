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

def load_images_from_folder(folder):
    images = None
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))[:,:,0]
        img = np.reshape(img,(1,-1))
        # print(img.shape)
        
        if img is not None:
            
            if images is None:
                images = img
                continue
                
            # print(images.shape)
            # print(img.shape)
            images = np.vstack((images, img))
            print(images.shape)
            
            
    return images


tf.autograph.set_verbosity(0)

folder0 = '/home/tingxfan/machine_learning/class_images/original_images_0/compressed'
folder1 = '/home/tingxfan/machine_learning/class_images/original_images_1/compressed'

# input_x = np.hstack((load_images_from_folder(folder0),load_images_from_folder(folder1)))
zeros = load_images_from_folder(folder0)
ones = load_images_from_folder(folder1)
X = np.vstack((zeros,ones))
Y = np.hstack((np.zeros((zeros.shape[0])),np.ones((ones.shape[0]))))

Xt = X
Yt = Y
print(Xt.shape)
print(Yt.shape)

layer_1 = Dense(units=250, activation="relu")
layer_2 = Dense(units=150, activation="relu")
layer_3 = Dense(units=15, activation="relu")
layer_4 = Dense(units=1, activation="sigmoid")

model = Sequential([tf.keras.Input(shape=(160000,)), layer_1, layer_2, layer_4])
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)



# checkpoint_path = '/home/tingxfan/machine_learning/class_images/ckpt/training1.ckpt'

checkpoint_path = "training_5/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))



# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=25
                                                 )

model.summary()

model.fit(
    Xt,Yt,            
    epochs=100,
    callbacks = cp_callback,
)

test_path = '/home/tingxfan/machine_learning/class_images/test_images/joshua-fuller-UJ7t9uNcTDU-unsplash.jpg'
test_img = load_iamge(test_path)
X_test = np.reshape(test_img, (1,-1))

show_image(test_img)

predictions = model.predict(X_test)
print("predictions = " + str(predictions))

result = model.evaluate(Xt, Yt)
print(result)
# print("Restored model, accuracy: {} %".format(100 * result))