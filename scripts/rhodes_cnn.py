#!/bin/usr/python

import sys
import os
import pickle
import numpy.matlib
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import convolutional, Merge, advanced_activations
from keras.optimizers import Adagrad
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
#import theano
#theano.config.device='gpu0'
#theano.config.floatX = 'float32'

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


longest_sentence_corpus = 0
merged_model = None
author_set = None

def center_sentence_tensor(tensor_file):
    curr_doc_tensor = pickle.load(open(tensor_file, 'rb'))
    num_sent, max_sent_len, word_dim = curr_doc_tensor.shape
    
    """
    print "num sent: " + str(num_sent) + "\n"
    print "max sent len: " + str(max_sent_len), word_dim
    print "word dim"

    """
    centered_tensor = np.zeros((num_sent, longest_sentence_corpus, word_dim))

    for sent_ind in xrange(num_sent - 1, -1, -1):
        # find the len of this sentence
        word_ind = max_sent_len - 1
        while word_ind >= 0 and sum(curr_doc_tensor[sent_ind, word_ind, :]) == 0:
            word_ind -= 1
        
            # bias towards having no padding instead of having some...
            pad_amount = (longest_sentence_corpus - word_ind)/2

        start_write = pad_amount + 1
        end_write = start_write + word_ind + 1
        start_read = 0
        end_read = word_ind + 1

        centered_tensor[sent_ind, start_write:end_write,:] = curr_doc_tensor[sent_ind, start_read:end_read, :]
    
    return centered_tensor


def get_longest_corpus_sentence(train_dir, test_dir):
    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)

    original_train = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-training-corpus-2012-03-28'
    original_test = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-test-corpus-2012-05-24'
    
    
    train_files = [x for x in sorted(os.listdir(original_train)) if x != "12Esample01.txt"
            and x != "12Fsample01.txt"
            and x != "README.txt"]
    test_files = [x for x in sorted(os.listdir(original_test)) if x != 'ground-truth.txt'
            and x != 'README.txt']

    #print train_files
    #print test_files
    '''
    global longest_sentence_corpus
    longest_sentence_corpus = 0
    for curr_train in train_files:
        tensor_obj = pickle.load(open(train_dir + '/' + curr_train))
        _, curr_sent_len, _ = tensor_obj.shape
        #global longest_sentence_corpus
        longest_sentence_corpus = max(longest_sentence_corpus, curr_sent_len)

    for curr_test in test_files:
        tensor_obj = pickle.load(open(test_dir + '/' + curr_test))
        _, curr_sent_len, _ = tensor_obj.shape
        #global longest_sentence_corpus
        longest_sentence_corpus = max(longest_sentence_corpus, curr_sent_len)
    '''

    longest = 0
    for curr_train in train_files:
        #print curr_train
        read_file = open(original_train + '/' + curr_train, 'r')
        curr_file_string = read_file.read().replace('\n', ' ').replace('\r', '')
        curr_file_string = unicode(curr_file_string, errors='replace')
        sent_segm = sent_tokenize(curr_file_string)

        for sent in sent_segm:
            longest = max(len(word_tokenize(sent)), longest)
    
    for curr_test in test_files:
        #print curr_test
        read_file = open(original_test + '/' + curr_test, 'r')
        curr_file_string = read_file.read().replace('\n', ' ').replace('\r', '')
        curr_file_string = unicode(curr_file_string, errors='replace')
        sent_segm = sent_tokenize(curr_file_string)

        for sent in sent_segm:
            longest = max(len(word_tokenize(sent)), longest)

    global longest_sentence_corpus
    longest_sentence_corpus = longest
    print "done getting longest"


#def sanity_check(doc_path):
#    read_file = open(doc_path, 'r')
#    curr_file_string = read_file.read().replace('\n', ' ').replace('\r', '')
#    curr_file_string = unicode(curr_file_string, errors='replace')
#    sent_segm = sent_tokenize(curr_file_string)
#    #print sent_segm
#
#    longest = 0
#
#    for sent in sent_segm:
#        longest = max(len(word_tokenize(sent)), longest)

#    print "longest: " + str(longest) + ", num sentences: " + str(len(sent_segm))



def build_rhodes():
    print "building rhodes"
    auth_model_3gram = Sequential()
    auth_model_3gram.add(convolutional.Convolution1D(100, 3, border_mode='same', input_shape=(longest_sentence_corpus, 300)))
    auth_model_3gram.add(Activation('relu'))
    #auth_model_3gram.add(convolutional.Convolution1D(100, 3, border_mode='same', input_shape=(longest_sentence_corpus, 300)))
    #auth_model_3gram.add(convolutional.MaxPooling1D(pool_size=(1,300),strides=None, border_mode='same', dim_ordering='th'))    
    auth_model_3gram.add(convolutional.MaxPooling1D(2, stride=1, border_mode='same')) 
    auth_model_3gram.add(Dropout(0.25)) 

    auth_model_4gram = Sequential()
    #auth_model_4gram.add(convolutional.Convolution2D(100, 4, 300, border_mode='same', input_shape=(longest_sentence_corpus, 300)))
    #auth_model_4gram.add(convolutional.MaxPooling2D(pool_size=(1,300),strides=None, border_mode='same', dim_ordering='th'))    
    auth_model_4gram.add(convolutional.Convolution1D(100, 4, border_mode='same', input_shape=(longest_sentence_corpus, 300)))
    auth_model_4gram.add(Activation('relu'))
    auth_model_4gram.add(convolutional.MaxPooling1D(2, stride=1, border_mode='same'))   
    auth_model_4gram.add(Dropout(0.25))

    auth_model_5gram = Sequential()
    #auth_model_5gram.add(convolutional.Convolution2D(100, 5, 300, border_mode='same', input_shape=(longest_sentence_corpus, 300))) 
    #auth_model_5gram.add(convolutional.MaxPooling2D(pool_size=(1,300),strides=None, border_mode='same', dim_ordering='th'))    
    auth_model_5gram.add(convolutional.Convolution1D(100, 5, border_mode='same', input_shape=(longest_sentence_corpus, 300)))
    auth_model_5gram.add(Activation('relu'))
    auth_model_5gram.add(convolutional.MaxPooling1D(2, stride=1, border_mode='same'))   
    auth_model_5gram.add(Dropout(0.25))

    global merged_model
    merged_model = Sequential()
    # addition
    #merged_model.add(convolutional.Convolution1D(100,3,border_mode='same', input_shape=(longest_sentence_corpus, 300)))
    #merged_model.add(Activation('relu'))
    #merged_model.add(convolutional.MaxPooling1D(2, stride=1, border_mode='same'))

    merged_model.add(Merge([auth_model_3gram, auth_model_4gram, auth_model_5gram], mode='concat', concat_axis=2))
    merged_model.add(Flatten())
    merged_model.add(Dense(200, activation='relu'))
    merged_model.add(Dropout(0.2))
    merged_model.add(Dense(len(author_set), activation='softmax'))
    
    merged_model.summary()
    
    ada = Adagrad(lr=0.01, epsilon=1e-06)
    merged_model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])
    
    
def get_author_labels(train_dir, test_dir):
    train_set = set()
    test_set = set()

    for file_name in os.listdir(train_dir):
        print file_name.split('train')
        train_set.add(file_name.split('train')[1][0])

    for file_name in os.listdir(test_dir):
        test_set.add(file_name.split('test')[1][0])

    assert len(train_set - test_set)
    global author_set
    author_set = train_set

def train_tensor(doc_tensor_path):
    # 1 hot vector encoding for categorical cross entropy loss (multinomial logistic regression)
    
    file_name = doc_tensor_path.split('/')[-1]
    

    #print doc_tensor_path.split('train')[1][0]
    #print str(ord(doc_tensor_path.split('n')[1][0]))
    author_index = ord(file_name.split('n')[1][0]) - 65
    label_vector = np.zeros((1,len(author_set)))
    label_vector[author_index] = 1.0
    
    doc_tensor = center_sentence_tensor(doc_tensor_path)
    num_sent, max_sent_len, word_dim = doc_tensor.shape
    label_matrix = np.matlib.repmat(label_vector, num_sent, 1)
   
    history = LossHistory()
    global merged_model

    print "doc tensor shape: " + str(doc_tensor.shape)
    

    for curr_sent_ind in xrange(0, num_sent):
        reshape_tensor = np.zeros((1, max_sent_len, word_dim))
        reshape_tensor[0, :, :] = doc_tensor[curr_sent_ind,:,:]
        print "longest sentence in corpus: " + str(longest_sentence_corpus)
        print "reshape tensor shape: " + str(reshape_tensor.shape)
        print "label vector: " + str(label_vector.shape)
        print "label matrix shape: " + str(label_matrix.shape)
        print type(label_matrix)
        print str(label_matrix)
        
        history = merged_model.fit([reshape_tensor, reshape_tensor, reshape_tensor], label_vector, verbose=1)     
    
    merged_model.save_weights('naive_run.h5')
    json_string = model.to_json()
    print json_string
    print history.losses
 
if __name__ == "__main__":
    get_author_labels(sys.argv[1], sys.argv[2])
    get_longest_corpus_sentence(sys.argv[1], sys.argv[2])
    build_rhodes()
    train_tensor(sys.argv[3])
   
