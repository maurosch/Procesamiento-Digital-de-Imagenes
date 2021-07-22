function z = sigmoid(x)
    z = 1./(1+exp(-x));
    %dSigmoid = z * (1 - z);
end