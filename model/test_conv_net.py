from convNet import ConvNet
import utils as utl

# get train inputs and targets
x, t = utl.sample_pos_neg_equal(5)
print((x.shape))
print(t.shape)

# Intialize network
convNet = ConvNet((256, 256, 3), epochs=50)

convNet.fit(x, t)

convNet.visualize_perf()



