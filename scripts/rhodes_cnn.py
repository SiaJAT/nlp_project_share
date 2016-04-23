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


W2V_train = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_train_WORD2VEC'
W2V_test = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_test_WORD2VEC'
pan_original_train = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-training-corpus-2012-03-28'
pan_original_test = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-test-corpus-2012-05-24' 
pan_test_labels_path = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-test-corpus-2012-05-24/labels.txt'

sample_train = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_train_WORD2VEC/12AtrainA1_tensor.p'
sample_test = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_test_WORD2VEC/12Atest01_tensor.p'


longest_sentence_corpus = 0
merged_model = None
author_set = None
pan_test_labels = {}

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


def get_longest_corpus_sentence(original_train, original_test): 
    train_files = [x for x in sorted(os.listdir(original_train)) if x != "12Esample01.txt"
            and x != "12Fsample01.txt"
            and x != "README.txt"]
    test_files = [x for x in sorted(os.listdir(original_test)) if x != 'ground-truth.txt'
            and x != 'README.txt']
 
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


def build_rhodes():
    print "building rhodes"
    auth_model_3gram = Sequential()
    auth_model_3gram.add(convolutional.Convolution1D(100, 3, border_mode='same', input_shape=(longest_sentence_corpus, 300)))    
    auth_model_3gram.add(convolutional.MaxPooling1D(2, stride=1, border_mode='same')) 

    auth_model_4gram = Sequential()
    auth_model_4gram.add(convolutional.Convolution1D(100, 4, border_mode='same', input_shape=(longest_sentence_corpus, 300)))
    auth_model_4gram.add(convolutional.MaxPooling1D(2, stride=1, border_mode='same'))   

    auth_model_5gram = Sequential()
    auth_model_5gram.add(convolutional.Convolution1D(100, 5, border_mode='same', input_shape=(longest_sentence_corpus, 300)))
    auth_model_5gram.add(convolutional.MaxPooling1D(2, stride=1, border_mode='same'))   

    global merged_model
    merged_model = Sequential()
    merged_model.add(Merge([auth_model_3gram, auth_model_4gram, auth_model_5gram], mode='concat', concat_axis=2))
    merged_model.add(Flatten())
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

    # get the PAN test labels
    read_pan_test_labels(pan_test_labels_path)
    for test_file, test_author in pan_test_labels.iteritems():
        print test_file + ", " + test_author
        test_set.add(test_author)
    

    assert len(test_set - train_set) >= 0
    global author_set
    author_set = train_set
    print str(author_set)

def train_tensor(doc_tensor_path):
    # 1 hot vector encoding for categorical cross entropy loss (multinomial logistic regression)
    file_name = doc_tensor_path.split('/')[-1]
    
    # create author vector (ASSUME that num author determined: get_author_labels())
    author_index = ord(file_name.split('n')[1][0]) - 65
    label_vector = np.zeros((1,len(author_set)))
    label_vector[author_index] = 1.0
    
    doc_tensor = center_sentence_tensor(doc_tensor_path)
    num_sent, max_sent_len, word_dim = doc_tensor.shape
   
    history = LossHistory()
    global merged_model
 
    for curr_sent_ind in xrange(0, 1):
        reshape_tensor = np.zeros((1, max_sent_len, word_dim))
        reshape_tensor[0, :, :] = doc_tensor[curr_sent_ind,:,:] 
        
        history = merged_model.fit([reshape_tensor, reshape_tensor, reshape_tensor], label_vector, verbose=1)     


def batch_train(train_dir):
    train_pickles = os.listdir(train_dir)
    for f in train_pickles:
        train_tensor(train_dir + '/' + f)


def test_tensor(doc_tensor_path, corpus_id):
    file_name = doc_tensor_path.split('/')[-1]
    
    file_name_no_suffix = file_name.split('_')[0]
    
    print "no suffix: " + file_name_no_suffix
    # error: test file has no label
    if corpus_id is "PAN" and file_name_no_suffix not in pan_test_labels:
        print "file not found"
        return 
    
    # get the document tensor
    doc_tensor = center_sentence_tensor(doc_tensor_path)
    num_sent, max_sent_len, word_dim = doc_tensor.shape
    
    # get the author of this document
    author_index = pan_test_labels[file_name_no_suffix]
    author_index = ord(author_index.strip()) - 65

    # create author vector (ASSUME that num author determined: get_author_labels())
    label_vector = np.zeros((1,len(author_set)))
    label_vector[:,author_index] = 1.0

    # create a label vector for each sentence and store in a matrix
    label_mat = numpy.matlib.repmat(label_vector, num_sent, 1)
   

    # evaluate 
    score = merged_model.evaluate([doc_tensor, doc_tensor, doc_tensor], label_mat, verbose=0)
    print file_name_no_suffix + "," + str(score[0]) + "," + str(score[1])


def batch_test(test_dir, corpus_id):
    test_pickles = os.listdir(test_dir)
    for f in test_pickles:
        test_tensor(test_dir + '/' + f)

def read_pan_test_labels(test_labels_path):
    global pan_test_labels

    # labels file format
    # doc name (w/o suffix), author as capital letter
    with open(test_labels_path, 'r') as open_file:
        for line in open_file:
            #print line
            entry = line.split(',') 
            file_name, author = entry[0], entry[1].strip()
            pan_test_labels[file_name] = author


def save_model():
    global merged_model
    merged_model.save_weights('naive_run.h5')
    json_string = merged_model.to_json()
    open('naive_run.json','w').write(json_string)

def make_clean():
    if os.path.isfile('naive_run.h5'):
        os.remove('naive_run.h5')
    if os.path.isfile('naive_run.json'):
        os.remove('naive_run.json')


if __name__ == "__main__":
    # preliminaries (get all possible authors and longest sentence)
    make_clean()
    get_author_labels(W2V_train, W2V_test)
    get_longest_corpus_sentence(pan_original_train, pan_original_test)
    read_pan_test_labels(pan_test_labels_path)
    
    # build the net and test 
    build_rhodes()
    batch_train(W2V_train)
    train_tensor(sample_train)
    save_model()
    #test_tensor(sample_test, 'PAN')
    batch_test(W2V_test)
    '''
    dict = read_pan_test_labels(pan_test_labels_path)
    for k,v in dict.iteritems():
        print str(k) + "," + str(v)
    test_tensor(sample_test, 'PAN')
    get_author_labels(W2V_train, W2V_test)
    get_longest_corpus_sentence(pan_original_train, pan_original_test)
    test_tensor(sample_test, 'PAN')
    '''
