import numpy as np
def samplePatches(rawImages, patchSize, numPatches):
    # rawImages is of size (image width)*(image height) by number_of_images
    # We assume that image width = image height
    imWidth = np.sqrt(np.size(rawImages,1));
    imHeight = imWidth;
    numImages = np.size(rawImages,2);
    rawImages = np.reshape(rawImages,imWidth,imHeight,numImages); 

    # Initialize patches with zeros.  
    patches = np.zeros(patchSize*patchSize, numPatches);

    # Maximum possible start coordinate
    maxWidth = imWidth - patchSize + 1;
    maxHeight = imHeight - patchSize + 1;

    # Sample!
    for num in range(1,numPatches):
        x = np.random.randint(maxHeight);
        y = np.random.randint(maxWidth);
        img = np.random.randint(numImages);
        p = rawImages[x:x+patchSize-1,y:y+patchSize-1, img];
        patches[:,num] = p[:];
    
    return patches