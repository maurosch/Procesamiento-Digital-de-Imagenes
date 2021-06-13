import numpy as np


def sech(x):
    return 1 / np.cosh(x)

def activation(x):
    b = 2.5
    f	= np.tanh(b*x)
    df	= b*sech(b*x)**2
    return f, df

# La entrada consiste en un array cuyas columnas son los ejemplos de
# aprendizaje y cuyas lineas son los descriptores
X = np.array([[1, 1, -1, -1],[1, -1, 1, -1]])
Ni, M          = np.shape(X)
No		         = 1
Nh               = 2
eta              = 0.25;  # ratio de aprendizaje
epochs           = 10; # son las iteraciones para el aprendizaje
# Las salidas esperadas o targets
T = np.array([-1, 1, 1, -1])

# Se inicializa la red
# Hay una sola neurona de salida, con lo cual entre las neuronas escondidas
# y la salida hay un vector de pesos. Entre la entrada y las neuronas
# escondidas hay una matriz de pesos.

# Hidden weights (NhxNi)
Wh = np.array([[0.3615, -1.4145], [-0.8916, 0.2010]])
#Output weights (NoxNh)
Wo = np.array([-1.1678 -0.2166])
# Los bias se fijan a un valor igual a 1. No hay aprendizaje en ellos.
bo = 1
bh = 1

J = [0]*(epochs+1)
J[1]    = 1e3
m       = 0
while m < epochs:
    m = m + 1
    for i in range(1,M):
        # voy a tomar uno a uno los puntos para actualizar los pesos
        Xm = X[:,i]
        tk = T[i]
        # Forward propagation desde la entrada X
        # Calcular primero las aj en las neuronas escondidas
        aj = bo + Xm * tk; ## ESCRIBIR EL CODIGO AQUI - #0: [-0.0530 0.3094]
        y, dfh = activation(aj)
        # Calcular ahora el valor de salida utilizando lo precedente
        ak	= 0; ## ESCRIBIR EL CODIGO AQUI - #0: 1.0133
        zk, dfo = activation(ak)
        # Ya se puede calcular la salida y el error cometido
        # Evaluar ahora el delta_k a la salida: delta_k = (tk-zk)*f'(ak)
        delta_k	= 0 ## ESCRIBIR EL CODIGO AQUI - #0: -0.1238
        #...y delta_j: delta_j = f'(aj)*w_j*delta_k
        delta_j	= np.multiply(dfh, Wo) * delta_k ## ESCRIBIR EL CODIGO AQUI - #0: [0.3550 0.0388]
        # Ahora se actualizan los pesos
        #w_kj <- w_kj + eta*delta_k*y_j
        Wo	= Wo + eta*delta_k*y ## ESCRIBIR CODIGO AQUI
        #w_ji <- w_ji + eta*delta_j*Xm
        Wh	= Wh + eta*np.multiply(delta_j, Xm) ## ESCRIBIR CODIGO AQUI
    
    #Calcular el error total
    J[m]    = 0
    for i in range(1,M):
        J[m] = J[m] + (T[i] - activation(Wo*activation(Wh*X[:,i] + bh*np.ones((Ni,1))) + bo))**2
    
    J[m] = J[m]/M; 
    print('Iteracion ' + str(m) + ': Error Total ' + str(J[m]))

# El error a la salida debe ser aproximadamente 0.00071965
expErr = 0.00071965
assert(abs(J[epochs]-expErr) < 1e-6) #, 'Error de implementacion'

# Resultado
for i in range(1,M):
    test_res = activation(Wo*activation(Wh* X[:,i] + bh*np.ones((Ni,1))) + bo)
    print('X: ['+ str(X(1,i)) +','+ str(X(2,i)) +'] -> Esperado : '+ str(T(i)) +' - Calculado: '+ str(test_res))
