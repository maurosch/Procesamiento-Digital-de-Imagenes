import numpy as np
import imageio
def loadLPRImages(imageDim):

    print('Lectura de la base de caracteres');

    folder = np.fullfile(np.fullfile('svm_smo','SVMCode','Datasets','BaseOCR_MultiStyle'));

    dchars = {
        '0';
        '1';
        '2';
        '3';
        '4';
        '5';
        '6';
        '7';
        '8';
        '9'}';


    Base = [];
    Labels = [];

    for d in range(1,len(dchars)):
        # get file list
        files = dir(fullfile(folder,dchars{d},'*.bmp'));
        files = [[files], [dir(fullfile(folder,dchars{d},'*.tif'))]];
        for f in range(1,len(files)):
            filename = np.fullfile(folder,dchars{d},files(f).name);
            Ip = float(imageio.imread(filename));
            # redimensiono el caracter
            Ip = imageio.imresize(Ip,[imageDim, imageDim]);
            # normalizo entre 0 y 1
            Ip = Ip - min(Ip[:]);
            Ip = Ip / max(Ip[:]);
            
            Base = [Base, Ip[:]];      
            Labels = [Labels, d-1];

    return Base,Labels

