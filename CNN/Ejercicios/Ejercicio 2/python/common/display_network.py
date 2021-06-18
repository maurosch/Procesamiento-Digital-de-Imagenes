import numpy as np
import matplotlib.pyplot as plt
def display_network(A, opt_normalize=True, opt_graycolor=True, cols=None, opt_colmajor=False):
    # This function visualizes filters in matrix A. Each column of A is a
    # filter. We will reshape each column into a square image and visualizes
    # on each cell of the visualization panel. 
    # All other parameters are optional, usually you do not need to worry
    # about it.
    # opt_normalize: whether we need to normalize the filter so that all of
    # them can have similar contrast. Default value is true.
    # opt_graycolor: whether we use gray as the heat map. Default is true.
    # cols: how many columns are there in the display. Default value is the
    # squareroot of the number of columns in A.
    # opt_colmajor: you can switch convention to row major for A. In that
    # case, each row of A is a filter. Default value is false.
    #warning off all


    # rescale
    A = A - np.mean(A[:]);

    #if opt_graycolor, colormap(gray); end

    # compute rows, cols
    [L, M]=np.size(A);
    sz=np.sqrt(L);
    buf=1;
    if cols == None:
        if np.floor(np.sqrt(M))^2 != M:
            n=np.ceil(np.sqrt(M));
            while np.mod(M, n) != 0 and n<1.2*np.sqrt(M):
                n=n+1;
            m=np.ceil(M/n);
        else:
            n=np.sqrt(M);
            m=n;
        
    else:
        n = cols;
        m = np.ceil(M/n);

    array = -np.ones(buf+m*(sz+buf),buf+n*(sz+buf));

    if ~opt_graycolor:
        array = 0.1 * array;


    if ~opt_colmajor:
        k=1;
        for i in range(1,m):
            for j in range(1,n):
                if k>M: 
                    continue; 
                
                clim=max(abs(A[:,k]));
                if opt_normalize:
                    array(buf+(i-1)*(sz+buf)+list(range(1,sz)),buf+(j-1)*(sz+buf)+list(range(1,sz))) = np.reshape(A[:,k],sz,sz)/clim;
                else:
                    array(buf+(i-1)*(sz+buf)+list(range(1,sz)),buf+(j-1)*(sz+buf)+list(range(1,sz))) = np.reshape(A[:,k],sz,sz)/max(abs(A[:]));
                
                k=k+1;

    else:
        k=1;
        for j in range(1,n):
            for i in range(1,m):
                if k>M: 
                    continue; 
                
                clim=max(abs(A[:,k]));
                if opt_normalize:
                    array(buf+(i-1)*(sz+buf)+list(range(1,sz)),buf+(j-1)*(sz+buf)+list(range(1,sz))) = np.reshape(A[:,k],sz,sz)/clim;
                else:
                    array(buf+(i-1)*(sz+buf)+list(range(1,sz)),buf+(j-1)*(sz+buf)+list(range(1,sz))) = np.reshape(A[:,k],sz,sz);
                
                k=k+1;
 

 
    h = plt.imshow(array)
    plt.show()
    
    #axis image off

    #drawnow;

    #warning on all

    return [h, array]
