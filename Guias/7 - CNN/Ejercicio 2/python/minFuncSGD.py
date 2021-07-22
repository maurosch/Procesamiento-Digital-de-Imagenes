import numpy as np
def minFuncSGD(funObj,theta,data,labels, options):
    # Runs stochastic gradient descent with momentum to optimize the
    # parameters for the given objective.
    #
    # Parameters:
    #  funObj     -  function handle which accepts as input theta,
    #                data, labels and returns cost and gradient w.r.t
    #                to theta.
    #  theta      -  unrolled parameter vector
    #  data       -  stores data in m x n x numExamples tensor
    #  labels     -  corresponding labels in numExamples x 1 vector
    #  options    -  struct to store specific options for optimization
    #
    # Returns:
    #  opttheta   -  optimized parameter vector
    #
    # Options (* required)
    #  epochs*     - number of epochs through data
    #  alpha*      - initial learning rate
    #  minibatch*  - size of minibatch
    #  momentum    - momentum constant, defualts to 0.9


    ##======================================================================
    ## Setup
    assert('epochs' not in options or 'alpha' not in options or 'minibatch' not in options, 'Some options not defined');
    if 'momentum' not in options:
        options.momentum = 0.9;
  
    epochs = options.epochs;
    alpha = options.alpha;
    minibatch = options.minibatch;
    m = len(labels); # training set size
    # Setup for momentum
    mom = 0.5;
    momIncrease = 20;
    velocity = np.zeros(np.size(theta));

    ##======================================================================
    ## SGD loop
    it = 0;
    for e in range(1,epochs):
        
        # randomly permute indices of data for quick minibatch sampling
        rp = np.randperm(m);
        
        for s in range(1,minibatch,(m-minibatch+1)):
            it = it + 1;

            # increase momentum after momIncrease iterations
            if it == momIncrease:
                mom = options.momentum;
         

            # get next randomly selected minibatch
            mb_data = data[:,:,rp[s:s+minibatch-1]];
            mb_labels = labels[rp[s:s+minibatch-1]];

            # evaluate the objective function on the next minibatch
            cost, grad = funObj(theta,mb_data,mb_labels);
            
            # Instructions: Add in the weighted velocity vector to the
            # gradient evaluated above scaled by the learning rate.
            # Then update the current weights theta according to the
            # sgd update rule
            
            ### YOUR CODE HERE ###
            velocity = mom * velocity + alpha * grad;
            theta = theta - velocity;

            print(f'Epoch {e}: Cost on iteration {it} is {cost}#f\n');
        

        # aneal learning rate by factor of two after each epoch
        alpha = alpha/2.0;



    opttheta = theta;
    return opttheta


