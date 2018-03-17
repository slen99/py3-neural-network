# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 00:51:10 2018

@author: Shinelon
"""

from network3 import Network
from mnist_loader import load_data_wrapper


def main():
    training_data, validation_data, test_data = load_data_wrapper()

    net = Network([784, 50, 30, 10], 10)
    net.SGD(list(training_data), 50, 10, 4.0, list(validation_data), list(test_data))


if __name__ == "__main__":
    main()
