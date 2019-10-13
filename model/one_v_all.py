import numpy as np
from convNet import ConvNet
import utils as utl
from sklearn.model_selection import train_test_split


# get train inputs and targets
x, t = utl.sample_pos_neg_equal(50, sample_size=10)
print(x.shape)
print(t.shape)
x, x_val, t, t_val = train_test_split(x, t, test_size=0.05)

convNet = ConvNet(input_shape=(256, 256, 3),
                  learning_rate=0.0001,
                  kernel_size=[3, 3],
                  epochs=3,
                  batch_size=30)

convNet.fit(x, t, [x_val, t_val])

convNet.visualize_perf()

convNet.evaluate_model(x_val, t_val)

pred = convNet.predict(x_val)
for i in range(0, len(t_val)):
    print('------------------------------')
    print('label: {} pred: {}'.format(t_val[i], pred[i]))
    utl.display_image(x_val[i])
