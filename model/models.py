import numpy as np
from convNet import ConvNet
import utils as utl


for i in range(53, 101):
    x, t = utl.sample_pos_neg_equal(i, sample_size=10)
    convNet = ConvNet(input_shape=(256, 256, 3),
                      learning_rate=0.001,
                      kernel_size=[3, 3],
                      epochs=3,
                      batch_size=30)
    convNet.fit(x, t)
    convNet.model_save(i)
