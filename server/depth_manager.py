import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import pdb
import pandas as pd

import models

def predict(model_data_path, img, showPlot=False):

    tf.reset_default_graph()
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result
        if showPlot:
            ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
            
        
        return pred

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)
    
    os._exit(0)

def estimate_volume(img):
    thr = 1.05
    model_path = "NYU_FCRN.ckpt"
    preds = t_preds = predict(model_path,img=img, showPlot=False)
    t_preds[t_preds<thr] = 1
    t_preds[t_preds>=thr] = 0
    return t_preds.sum()/t_preds.size
    
def estimate_cal(img,label):
    CONSTANT = 6
    vol = estimate_volume(img)
    map_list = pd.read_csv("..\data\labels_online.csv")
    cal_p_g = map_list[map_list.Name==label].Energy.iloc[0]
    return vol*cal_p_g*CONSTANT
    
    C:\Users\Stefan\Documents\Hackathon\HACK_UPC\data
    


