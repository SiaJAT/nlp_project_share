import sys
import os
import subprocess as sp
import pprint
import random
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pickle

class InvertedIndex:
    
    '''
    an inverted index with the following structure:
    
    {word : {doc : [(s_1, [w_1, ..., w_n])]}}
    
    a dictionary from words to a list of tuples, 
    where each tuple maps from a sentence index 
    '''
    def __init__(self):
        self.word_2_docSentList = {}
        

    def insert(self, word, doc, sent_num, word_pos):
        # this word has been seen before
        if word in self.word_2_docSentList:
            doc2sent_num = self.word_2_docSentList[word]
            
            # this word in this document has been seen before
            if doc in doc2sent_num:
                sent_num2word  = doc2sent_num[doc]
                
                # this word has been seen in this document in this same sentence
                if sent_num in sent2word:
                    word_indices = sent_num2word[sent_num]
                    word_indices.append(word_pos)
                    sent_num2word[sent_num] = word_indices
                
                # this word in this document has been seen in a different sentence
                else:
                    sent_num2word[sent_num] = [word_pos]

            # this word has not been seen in this document before
            else:
                doc2sent_num[doc] = {sent : [word_pos]}
            
        # first time the word has been seen in the corpus
        else:
            sent2word = {sentence_num: [word_pos]}
            doc2sent = {doc: sent2word}
            self.word_2_docSentList[word] = doc2sent
            
    
    def build_representation(self, curr_dir):
        
        reload(sys)
        sys.setdefaultencoding('utf8')

        # list all of the words in the directory 
        curr_dir_files = [x for x in sorted(os.listdir(curr_dir)) if x is not "12Esample01.txt" 
                and x is not "12Fsample01.txt" 
                and x is not "README.txt"]

        #print curr_dir_files

        # serialize pretrained docs written to "glove_dict.p"
        # presently hard coded 
        #serialize_pretrained_vecs_data('/mnt0/siajat/cs388/nlp/data/glove_sample.txt', "glove")

        #word_dim = len(word_list)
         
        # save the old directory and go to where the
        # other files are stored
        old_dir = os.getcwd()
        os.chdir(curr_dir)
     

        for f in curr_dir_files:
            doc_name = f.split('.')[0]
            
            with open(f, 'r') as curr_file:
                # clean the string
                curr_file_string = curr_file.read().replace('\n', ' ').replace('\r', '')
                curr_file_string = unicode(curr_file_string, errors='replace')
                
                # segment the sentence 
                sent_segm = sent_tokenize(curr_file_string)
                num_sentences = len(sent_segm)
                max_sent_len = len_longest_sentence(curr_file_string)
         
                # tokenize each sentence
                sent_counter = 0
                for sent in sent_segm:
                    tok_segm = word_tokenize(sent)
                    
                    tok_counter = 0 
                    for tok in tok_segm:   
                        self.insert(word, doc_name, sent_counter, tok_counter)
                        tok_counter += 1    
                    
                    sent_counter += 1
                
                #pickle.dump(save_arr, open(old_dir + '/' + doc_name + '_npArr.p','wb'))
    
    
    def len_longest_sentence(self, curr_file_string):
        sent_segm = sent_tokenize(curr_file_string)
    
        longest = 0

        for sent in sent_segm:
            longest = max(len(word_tokenize(sent)), longest)

        return longest

    
    def serialize_index(self, save_name):
        pickle.dump(self.word_2_docSentList, open(save_name + ".p", 'wb'))




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
   


if __name__ == "__main__": 
    #print "hello"
    #build_representation(sys.argv[1])
    #serialize_pretrained_vecs_data('/mnt0/siajat/cs388/nlp/data/glove.840B.300d.txt', "glove_complete")
    #index = InvertedIndex()
    #index.build_representation(sys.argv[1])
