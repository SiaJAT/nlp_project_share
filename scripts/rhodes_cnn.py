#!/bin/usr/python

import pickle
from keras.layers import convolutional
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


def centering(tensor_file):
    curr_doc = pickle.load(open(tensor_file, 'rb'))
    a, b, c = curr_doc.shape
    print str(a) + ", " + str(b) + ", " + str(c)

def sanity_check(doc):


def build_rhodes():
    language_model = Sequential()
    language_model.add(convolutional.Convolution2D())


if __name__ == "__main__":
    centering(sys.argv[1])
