import numpy as np
import cv2 as cv

path = 'Dataset/training_data/'

def load_data(min, max):
    imgs = []
    for img_file in range(min, max):
        imgs.append(cv.imread(path + str(img_file) + '.jpg'))

    return imgs


images = load_data(0, 5)

cv.imshow('image', images[0])


