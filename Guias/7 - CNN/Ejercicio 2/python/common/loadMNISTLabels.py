import numpy as np


def loadMNISTLabels(filename):
    #loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
    #the labels for the MNIST images

    all = np.fromfile(filename,dtype='>i4',count=2)
    magic = all[0];
    assert magic == 2049, f'Bad magic number in {filename}'

    numLabels = all[1];

    labels = np.fromfile(filename,dtype=np.ubyte,offset=2*4)

    assert np.size(labels) == numLabels, 'Mismatch in label count';

    return labels
