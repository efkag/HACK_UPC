import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

imgs_path = 'Dataset/training_data/'
labels_path = 'Dataset/training_labels.txt'


def load_data_range(min, max):
    """
    Load images from the dataset within a given range
    :param min: Lower bound for array index
    :param max: Uper bound for array index
    :return: Python list of 3d numpy images
    """
    imgs = []
    for img_file in range(min, max):
        imgs.append(cv.imread(imgs_path + str(img_file) + '.jpg'))

    return imgs


def load_data_indexes(indexes):
    """
    Load image that are in the given index list
    :param indexes: List of indexes
    :return: List of images
    """
    imgs = []
    for img_file in indexes:
        imgs.append(cv.imread(imgs_path + str(img_file) + '.jpg'))

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


def load_labels():
    """
    Read labels into python list
    :return: Python list of labels
    """
    labels = []
    with open(labels_path, 'r') as f:
        for line in f:
            label = line.split()
            labels.append(label)
    return labels


def sample_rand_from_each_class(samples_num):
    """
    Sample a gicven number of images from each class
    :param samples_num: Samples per class
    :return: Full dataset sample (should be of size samples_num * 101)
    """
    data_pointer = 0
    interval = 1001
    dataset_sample = []
    for i in range(0, 101):
        class_sample = []
        # Sample indexes with the class range
        indexes = np.random.randint(data_pointer, data_pointer + interval, samples_num)
        # Load all images given by the indexes
        class_sample = load_data_indexes(indexes)
        # Append to the dataset sample
        dataset_sample.extend(class_sample)

    print(len(dataset_sample))
    return dataset_sample
    # sample_data = []
    # sample_data.append(np.random.choice(data))


# # Example to load and display images
# images = load_data(0, 5)
# display_image(images[0])
#
# # Example to load labels
# labels = load_labels()
# print(len(labels))

#Sample from all classes
sample_rand_from_each_class(20)
