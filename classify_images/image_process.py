import cv2
import os
import numpy as np


dsize = (400, 400)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            output = cv2.resize(gray_image, dsize)
            images.append(output)
            cv2.imwrite(os.path.join(folder, 'compressed', filename), output)
    return images

folder0 = 'original_images_0'
folder1 = 'original_images_1'

zeros = load_images_from_folder(folder0)
ones = load_images_from_folder(folder1)



# cv2.imshow('image', output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

