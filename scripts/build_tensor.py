#!/usr/bin/python

import numpy as np
import sys
import os
import subprocess as sp
from document_tensor import DocumentTensor

def batch_build_pan_train(output_dir, dir, inv_ind_file, model_type):
    files = [x.split('.')[0] for x in os.listdir(dir) 
            if x != "12Esample01.txt" and 
            x != "12Fsample01.txt" and
            x != "README.txt" and 
            "12E" not in x and
            "12F" not in x]

    for f in files:
        tens = DocumentTensor(f)
        tens.build(inv_ind_file, model_type)
        tens.serialize_tensor()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for f in files:
        file_name = f + '.p'
        sp.Popen(['mv', file_name, output_dir])


if __name__ == "__main__": 
    batch_build_pan_train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
