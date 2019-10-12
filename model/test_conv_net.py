import numpy as np
from convNet import ConvNet
import utils as utl
from sklearn.model_selection import train_test_split


# get train inputs and targets
x, t = utl.sample_pos_neg_multy(sample_size=400)
print(x.shape)
print(t.shape)
x, x_val, t, t_val = train_test_split(x, t, test_size=0.10)



# Intialize network
convNet = ConvNet(input_shape=(256, 256, 3),
                  kernel_size=[5, 5],
                  epochs=3)

convNet.fit(x, t, [x_val, t_val])

convNet.visualize_perf()

pred = convNet.predict(x_val)

for i in range(0, len(t_val)):
    #print('label: {} pred: {}'.format(t_val[i], pred[i]))
    print(np.argmax(t_val[i]))
    print(np.argmax(pred[i]))
    utl.display_image(x_val[i])






