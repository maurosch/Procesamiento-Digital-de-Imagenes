import numpy as np


def cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses):
       # Initialize parameters for a single layer convolutional neural
       # network followed by a softmax layer.
       #                            
       # Parameters:
       #  imageDim   -  height/width of image
       #  filterDim  -  dimension of convolutional filter                            
       #  numFilters -  number of convolutional filters
       #  poolDim    -  dimension of pooling area
       #  numClasses -  number of classes to predict
       #
       #
       # Returns:
       #  theta      -  unrolled parameter vector with initialized weights

       ## Initialize parameters randomly based on layer sizes.
       assert(filterDim < imageDim,'filterDim must be less that imageDim');

       Wc = 1e-1*np.random.randn(filterDim,filterDim,numFilters);

       outDim = imageDim - filterDim + 1; # dimension of convolved image

       # assume outDim is multiple of poolDim
       assert(np.mod(outDim,poolDim)==0,'poolDim must divide imageDim - filterDim + 1');

       outDim = outDim/poolDim;
       hiddenSize = outDim**2 * numFilters;

       # we'll choose weights uniformly from the interval [-r, r]
       r  = np.sqrt(6) / np.sqrt(numClasses+hiddenSize+1);
       Wd = np.random.rand(int(numClasses), int(hiddenSize)) * 2 * r - r;

       bc = np.zeros((numFilters, 1));
       bd = np.zeros((numClasses, 1));

       # Convert weights and bias gradients to the vector form.
       # This step will "unroll" (flatten and concatenate together) all 
       # your parameters into a vector, which can then be used with minFunc. 
       theta = np.concatenate((Wc.reshape(Wc.size,order='F'), Wd.reshape(Wd.size,order='F'), bc.reshape(bc.size,order='F'), bd.reshape(bd.size,order='F')));
       #print(theta.shape)
       #print(Wc.reshape(Wc.size,order='F').shape)
       #print(theta)

       return theta
