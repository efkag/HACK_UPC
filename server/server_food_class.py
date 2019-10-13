#python server
from flask import Flask, Response, request
from flask_restful import Api, Resource
import cv2
import numpy as np
import jsonpickle
from PIL import Image
import io
import base64
from io import BytesIO

import scipy.io
import os
from PIL import Image
import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import applications
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
import depth_manager as dm

from base64 import decodestring

app = Flask(__name__)
api = Api(app)


model = load_model('model5ep.h5')
graph = tf.get_default_graph()


lines = [line.rstrip('\n') for line in open('labels.txt')]

print(lines)


class ClassificationAPI1(Resource):
    @app.route('/api/image', methods=['POST'])
    def image_post() :
        # do classification

        r = request
        # convert string of image data to uint8
        #nparr = np.fromstring(r.data, np.uint8)
     
        # decode image
        #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
        
        #im = Image.open(BytesIO(base64.b64decode(r.data)))
        #cv2.imwrite('testimg.jpg', im)

        im = Image.open(BytesIO(base64.b64decode(r.data)))
        im = im.resize((64, 64), Image.BILINEAR)
        cvim = np.array(im) 
        cvim = np.expand_dims(cvim, 0)

        global graph
        with graph.as_default():
            classes = model.predict(cvim, batch_size=1)
            print(classes.argmax())

        im.save('image.png', 'PNG')
        maxprob = classes.argmax()
        classname = lines[maxprob]


        # code for classifying image and returning class
        #
        #
        #
        #
        volume = dm.estimate_cal(img=im,label=classname)



        # response from classification (example)
        foodname = classname
        calories_count = 300


        response = {'Food name': foodname}
        
        response_pickled = jsonpickle.encode(response)

        return Response(response=response_pickled, status=200, mimetype="application/json")
        
app.run(debug=False, threaded = False, host= '0.0.0.0')