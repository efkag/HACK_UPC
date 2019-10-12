import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

imgs_path = 'Dataset/training_data/'
labels_path = 'Dataset/training_labels.txt'
number_of_classes = 101

def load_data_range(min, max):
    """
    Load images from the dataset within a given range
    :param min: Lower bound for array index
    :param max: Uper bound for array index
    :return: Python list of 3d ndarray images
    """
    imgs = []
    for img_file in range(min, max):
        imgs.append(cv.imread(imgs_path + str(img_file) + '.jpg'))

    return imgs


def load_data_indexes(indexes):
    """
    Load image that are in the given index list
    :param indexes: List of indexes
    :return: List of ndarray images
    """
    imgs = []
    for img_file in indexes:
        imgs.append(cv.imread(imgs_path + str(img_file) + '.jpg'))

    return imgs


def display_image(image):
    """
    Display the image given as a 2d or 3d ndarray of values.
    :param image: Input image to display
    """
    image = np.squeeze(image)
    plt.imshow(image, interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # or plt.axis('off')
    plt.show()


def get_img_size(image):
    """
    Return the image shape if the image given is ndarray
    :param image: ndarray representation of the image
    """
    height, width = np.squeeze(image).shape
    return height, width


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
    interval = 1000
    dataset_sample = []
    for i in range(0, number_of_classes):
        class_sample = []
        # Sample indexes with the class range
        indexes = np.random.randint(data_pointer, data_pointer + interval, samples_num)
        # Load all images given by the indexes
        class_sample = load_data_indexes(indexes)
        # Append to the dataset sample
        dataset_sample.extend(class_sample)

    #print(len(dataset_sample))
    return dataset_sample


def sample_random_from_class(class_pointer, sample_num):
    """
    Sample from a given class a given number of samples
    :param class_pointer: An integer from [0-101] indicating the class
    :param sample_num: An integer number of the samples requested
    :return: List of ndarray sample images from the class
    """
    interval = 1000
    # Sample indexes with the class range
    indexes = np.random.randint(class_pointer, class_pointer + interval, sample_num)
    # Load all images given by the indexes
    class_sample = load_data_indexes(indexes)
    return class_sample



def sample_pos_neg_equal(pos_class):
    """
    Create dataset of 1000 positives and 1000 negatives with 10 negatives
    from each of the negative classes. Also returns targets
    :param pos_class: Int the number of the possitive class
    :return: List of dataset sample ndarray + targets
    """
    interval = 1000
    sample_size = 10
    dataset_sample = []

    # using int to string and back to int to get
    # the correct index for class indexes
    lower = str(pos_class)
    lower += '000'
    lower = int(pos_class)
    upper = lower + interval
    # TODO: Need to ask Norbert how manny classes there are and if the interval
    # 1000 images for each class
    # Sample full positive class
    dataset_sample.extend(load_data_range(lower, upper))
    # create positive targets
    targets = np.full((1000,), 1)

    # Sample 10 images from each of the other classes
    for i in range(0, number_of_classes):
        if i is not pos_class:
            dataset_sample.extend(sample_random_from_class(i, sample_size))

    targets = np.concatenate((targets, np.full((1000,), 0)))

    return np.array(dataset_sample), targets



# # Examples for each of the methods
# images = load_data(0, 5)
# display_image(images[0])
#
# Example to load labels
# labels = load_labels()
# print(len(labels))
# print(labels[0])
# print(labels[1000])
# labels = np.array(labels)
# labels = np.unique(labels)
# print(labels.shape)

# #Sample from all classes
# sample_rand_from_each_class(20)

# train_sample = sample_pos_neg_equal(2)
# print(len(train_sample))
