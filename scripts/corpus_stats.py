import matplotlib.pyplot as plt
import operator
import sys
import os
import pickle
import numpy.matlib
import numpy as np
from sklearn import preprocessing
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


def corpus_stats(original_train, original_test, max_allowable):
    train_files = [x for x in sorted(os.listdir(original_train)) if x != "12Esample01.txt"
            and x != "12Fsample01.txt"
            and x != "README.txt"]
    test_files = [x for x in sorted(os.listdir(original_test)) if x != 'ground-truth.txt'
            and x != 'README.txt']
    
    natural2count = {}
    synthetic2count = {}

    for curr_train in train_files:
        #print curr_train
        read_file = open(original_train + '/' + curr_train, 'r')
        curr_file_string = read_file.read().replace('\n', ' ').replace('\r', '')
        curr_file_string = unicode(curr_file_string, errors='replace')
        sent_segm = sent_tokenize(curr_file_string)
        
        for sent in sent_segm:
            tokenized = word_tokenize(sent)
                        
            if len(tokenized) <= max_allowable:
                if len(tokenized) not in natural2count:
                    natural2count[len(tokenized)] = 1
                else:
                    natural2count[len(tokenized)] += 1
            
            if len(tokenized) > max_allowable:
                super_sentence_count = len(tokenized) / max_allowable

                if max_allowable not in synthetic2count:
                    synthetic2count[max_allowable] = super_sentence_count
                else:
                    synthetic2count[max_allowable] += super_sentence_count
                
                remainder = len(tokenized) % max_allowable

                if remainder > 0:
                    if remainder not in synthetic2count:
                        synthetic2count[remainder] = 1
                    else:
                        synthetic2count[remainder] += 1
        
        read_file.close()

    for curr_test in test_files:
        #print curr_test
        read_file = open(original_test + '/' + curr_test, 'r')
        curr_file_string = read_file.read().replace('\n', ' ').replace('\r', '')
        curr_file_string = unicode(curr_file_string, errors='replace')
        sent_segm = sent_tokenize(curr_file_string)
        
        for sent in sent_segm:
            tokenized = word_tokenize(sent)
                        
            if len(tokenized) <= max_allowable:
                if len(tokenized) not in natural2count:
                    natural2count[len(tokenized)] = 1
                else:
                    natural2count[len(tokenized)] += 1
            
            if len(tokenized) > max_allowable:
                super_sentence_count = len(tokenized) / max_allowable

                if max_allowable not in synthetic2count:
                    synthetic2count[max_allowable] = super_sentence_count
                else:
                    synthetic2count[max_allowable] += super_sentence_count
                
                remainder = len(tokenized) % max_allowable

                if remainder > 0:
                    if remainder not in synthetic2count:
                        synthetic2count[remainder] = 1
                    else:
                        synthetic2count[remainder] += 1
        
        read_file.close()

    print "naturally occurring..."
    for key, val in natural2count.iteritems():
        print str(key) + ": " + str(val)
    
    print "synthetic..."
    for key, val in synthetic2count.iteritems():
        print str(key) + ": " + str(val)

if __name__ == "__main__":
    pan_original_train = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-training-corpus-2012-03-28'
    pan_original_test = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-test-corpus-2012-05-24' 
    
    corpus_stats(pan_original_train, pan_original_test, int(sys.argv[1]))
