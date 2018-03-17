# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 21:54:42 2018

@author: 沉默
"""
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt

from network import Network
from mnist_loader import load_data_wrapper


def main():
    training_data, validation_data, test_data = load_data_wrapper()
    epoch = 50
    net = Network([784, 40, 10])
    net.SGD(list(training_data), epoch, 20, 4.0, test_data = list(test_data))
    Epoch = np.arange(1,epoch+1)
    Accuracy = np.array(net.test_result)/100
    plt.plot(Epoch, Accuracy, '.')
    plt.show()


if __name__ == "__main__":
    main()
