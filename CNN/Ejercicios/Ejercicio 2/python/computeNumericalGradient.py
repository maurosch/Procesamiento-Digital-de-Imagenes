import numpy as np
def computeNumericalGradient(J, theta):
    # numgrad = computeNumericalGradient(J, theta)
    # theta: a vector of parameters
    # J: a function that outputs a real-number. Calling y = J(theta) will return the
    # function value at theta. 
    
    # Initialize numgrad with zeros
    numgrad = np.zeros(np.size(theta));

    # numgrad(i) is (the numerical approximation to) the 
    # partial derivative of J with respect to the i-th input argument, evaluated at theta.  
    #                
    # Compute the elements of numgrad one at a time. 

    epsilon = 1e-4;

    for i in range(1,len(numgrad)):
        oldT = theta[i];
        theta[i]=oldT+epsilon;
        pos = J[theta];
        theta[i]=oldT-epsilon;
        neg = J[theta];
        numgrad[i] = (pos-neg)/(2*epsilon);
        theta[i]=oldT;
        if np.mod(i,100)==0:
            print('Done with #d\n',i);

    return numgrad





## ---------------------------------------------------------------