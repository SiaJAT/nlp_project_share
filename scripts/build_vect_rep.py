import sys
import os
import subprocess as sp
import pprint
import random
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pickle

def build_representation(curr_dir):
    
    reload(sys)
    sys.setdefaultencoding('utf8')

    # list all of the words in the directory 
    curr_dir_files = [x for x in sorted(os.listdir(curr_dir)) if x is not "12Esample01.txt" and x is not "12Fsample01.txt" and x is not "README.txt"]

    print curr_dir_files

    # serialize pretrained docs written to "glove_dict.p"
    # presently hard coded 
    serialize_pretrained_vecs_data('/mnt0/siajat/cs388/nlp/data/glove_sample.txt', "glove")

    # read in the word dictionary
    dict = pickle.load(open('glove_dict.p','rb'))
    word_dim = len(dict.itervalues().next())
     
    # save the old directory and go to where the
    # other files are stored
    old_dir = os.getcwd()
    os.chdir(curr_dir)
 

    for f in curr_dir_files:
        prefix = f.split('.')[0]
        
        with open(f, 'r') as curr_file:
            # clean the string
            curr_file_string = curr_file.read().replace('\n', ' ').replace('\r', '')
            curr_file_string = unicode(curr_file_string, errors='replace')
            
            #print type(curr_file_string)

            # segment the sentence 
            sent_segm = sent_tokenize(curr_file_string)
            num_sentences = len(sent_segm)
            max_sent_len = len_longest_sentence(curr_file_string)
    
            # save the sentences
            save_arr = np.zeros((num_sentences, max_sent_len, word_dim))
            
            # tokenize each sentence
            sent_counter = 0
            for sent in sent_segm:
                tok_segm = word_tokenize(sent)
               
                mu, sigma = 0, 0.1
                tok_counter = 0
                for tok in tok_segm:
                    if tok in dict:
                        save_arr[sent_counter,tok_counter,:] = dict[tok]                
                    else:
                        save_arr[sent_counter,tok_counter,:] = np.random.normal(mu, sigma, word_dim)
                    tok_counter  += 1

                while tok_counter < max_sent_len:
                    save_arr[sent_counter,tok_counter,:] = np.random.normal(mu, sigma, word_dim)
                    tok_counter += 1
                
                sent_counter += 1
            

            pickle.dump(save_arr, open(old_dir + '/' + prefix + '_npArr.p','wb'))
                        
                        
                    
                

def serialize_pretrained_vecs_data(train_path, vec_type):
    file_len = 0
    with open(train_path, 'r') as read_file:
        for line in read_file:
            file_len += 1

    label2vec = {}

    with open(train_path, 'r') as read_file:
        for line in read_file:
            arr = line.split(' ') 

            curr_label = arr[0]
            curr_vec = arr[1:] 
            
            label2vec[curr_label] = np.array([float(x.strip()) for x in curr_vec])

    pickle.dump(label2vec, open(vec_type + "_dict.p", "wb"))


def len_longest_sentence(curr_file_string):
    sent_segm = sent_tokenize(curr_file_string)
    
    longest = 0

    for sent in sent_segm:
        longest = max(len(word_tokenize(sent)), longest)

    return longest
    


if __name__ == "__main__": 
    #build_representation(sys.argv[1])
    serialize_pretrained_vecs_data('/mnt0/siajat/cs388/nlp/data/glove.840B.300d.txt', "glove_complete")
