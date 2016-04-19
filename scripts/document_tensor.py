import numpy as np

class DocumentTensor:
    def __init(self, name):
        #File prefix name before extension. 
        self.name = name

        #Dimmensions for tensor. 
        self.num_sent = None
        self.max_sent_len = None
        self.word_dimmension = None

        #The tensor. 
        self.tensor = None

    def build(self, serialized_inverted_index_name):
        #Load pickled inverted index object.. 
        inverted_index = pickle.load(open(serialized_inverted_index_name, 'rb'))

        #Gets document inverted index dictionary as well as list of statistics for document. 
        word2sent, stats = inverted_index.get_doc_data(self.name)
        
        self.num_sent = stats[0]
        self.max_sent_len = stats[1]


        #Gets dimmensions for tensor and initializes it. 
        #TODO Get dimmensions from stats (also implement stats)
        self.tensor = np.zeros((self.num_sent, self.max_sent_len, self.word_dimmension))

        #Builds tensor by iterating over every word and every sentence. 
        for word in word2sent:

            sent2word = word2sent[word]

            for sent in sent2word:
                #Get list of sentence indeces where word appears. 
                word_index_list = sent2word[sent]

                #Place word vector where necessary in sentence matrix. 
                for word_index in word_index_list:
                    self.tensor[sent, word_index, :] = #TODO get numpy vector from db. 

