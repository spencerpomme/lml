# plot ad hoc CIFAR10 instances
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from scipy.misc import toimage

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# create a grid of 5x5 images:
for i in range(25):
    plt.subplot(330 + 1 + i)
    plt.imshow(toimage(X_train[i]))
# show the plot
plt.show()

