#!/usr/bin/python

import numpy as np
import sys
from document_tensor import DocumentTensor

if __name__ == "__main__":
    tens = DocumentTensor("12AtrainA1")
    tens.build("pan_train.p", "GLOVE")
    tens.serialize_tensor()
