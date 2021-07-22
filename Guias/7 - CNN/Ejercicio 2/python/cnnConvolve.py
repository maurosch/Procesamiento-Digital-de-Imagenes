import numpy as np


def cnnConvolve(filterDim, numFilters, images, Wc, bc):
    #  cnnConvolve Devuelve el resultado de hacer la convolucion de W y b con
    #  las imagenes de entrada
    #
    # Parametetros:
    #  filterDim - dimension del filtro
    #  numFilters - cantidad de filtros
    #  images - imagenes 2D para convolucionar. Estas imagenes tienen un solo
    #  canal (gray scaled). El array images es del tipo images(r, c, image number)
    #  Wc, bc - Wc, bc para calcular los features
    #         Wc tiene tamanio (filterDim,filterDim,numFilters)
    #         bc tiene tamanio (numFilters,1)
    #
    # Devuelve:
    #  convolvedFeatures - matriz de descriptores convolucionados de la forma
    #                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

    numImages = len(images[0][0]);
    imageDim = len(images[0]);
    convDim = imageDim - filterDim + 1;

    convolvedFeatures = np.zeros((convDim, convDim, numFilters, numImages));

    # Instrucciones:
    #   Convolucionar cada filtro con cada imagen para obtener un array  de
    #   tamaño (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
    #   de modo que convolvedFeatures(imageRow, imageCol, featureNum, imageNum) 
    #   es el valor del descriptor featureNum para la imagen imageNum en la
    #   region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
    #

    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            # convolucion simple de una imagen con un filtro
            convolvedImage = np.zeros((convDim, convDim));
            # Obtener el filtro (filterDim x filterDim) 
            filter = Wc[:,:,filterNum];

            # Girar la matriz dada la definicion de convolucion
            filter = np.rot90(np.squeeze(filter),2);
            # Obtener la imagen
            im = np.squeeze(images[:, :, imageNum]);

            ### IMPLEMENTACION AQUI ###
            # Convolucionar "filter" con "im", y adicionarlos a 
            # convolvedImage para estar seguro de realizar una convolucion
            # 'valida'
            convolvedImage = 0;

            # Agregar el bias 
            # Luego, aplicar la funcion sigmoide para obtener la activacion de 
            # la neurona.

            ### IMPLEMENTACION AQUI ###
            convolvedFeatures[:, :, filterNum, imageNum] = 0;

    return convolvedFeatures

