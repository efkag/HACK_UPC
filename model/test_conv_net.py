import numpy as np
from convNet import ConvNet
import utils as utl
from sklearn.model_selection import train_test_split


# get train inputs and targets
x, t = utl.sample_pos_neg_multi(sample_size=100)
print(x.shape)
print(t.shape)
x, x_val, t, t_val = train_test_split(x, t, test_size=0.05)

utl.display_image(x[0])
utl.print_class(t[0])


# Intialize network
convNet = ConvNet(input_shape=(256, 256, 3),
                  learning_rate=0.01,
                  kernel_size=[6, 6],
                  epochs=2,
                  batch_size=30)

convNet.fit(x, t, [x_val, t_val])

convNet.visualize_perf()

convNet.evaluate_model(x_val, t_val)

pred = convNet.predict(x_val)
print(len(x_val))
print(len(pred))

for i in range(0, len(t_val)):
    print('------------------------------')
    #print('label: {} pred: {}'.format(t_val[i], pred[i]))
    print(np.argmax(t_val[i]))
    print(np.argmax(pred[i]))
    utl.display_image(x_val[i])






