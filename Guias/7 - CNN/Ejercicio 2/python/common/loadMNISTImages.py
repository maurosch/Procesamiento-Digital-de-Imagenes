import numpy as np


def loadMNISTImages(filename):
    #loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
    #the raw MNIST images

    all = np.fromfile(filename,dtype='>i4',count=4)
    magic = all[0];
    assert magic == 2051, f'Bad magic number in {filename}';

    numImages = all[1];
    numRows = all[2];
    numCols = all[3];

    images = np.fromfile(filename,dtype=np.ubyte,offset=4*4)
    images = np.reshape(images, (numCols, numRows, numImages));
    #print(images.shape)
    images = np.transpose(images,(1, 0, 2));

    # Reshape to #pixels x #examples
    images = np.reshape(images, (np.size(images, 0) * np.size(images, 1), np.size(images, 2)));
    # Convert to double and rescale to [0,1]
    images = images.astype(float) / 255;

    return images
