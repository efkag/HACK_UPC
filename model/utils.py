import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

path = 'Dataset/training_data/'

def load_data(min, max):
    imgs = []
    for img_file in range(min, max):
        imgs.append(cv.imread(path + str(img_file) + '.jpg'))

    return imgs


def display_image(image):
    """
    Display the image given as a 2d or 3d array of values.
    :param image: Input image to display
    """
    image = np.squeeze(image)
    plt.imshow(image, interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # or plt.axis('off')
    plt.show()


def sample_from_each_class(data):
    sample_data = []
    sample_data.append(np.random.choice(data))



# Example to load and display images
images = load_data(0, 5)
display_image(images[0])
