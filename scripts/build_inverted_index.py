#!/usr/bin/python

import numpy
import sql
from inverted_index import InvertedIndex

if __name__ == "__main__":
    #Construct inverted index. 
    index = InvertedIndex()
    index.build_representation("/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-training-corpus-2012-03-28")
    index.serialize_inverted_index("pan_train")
