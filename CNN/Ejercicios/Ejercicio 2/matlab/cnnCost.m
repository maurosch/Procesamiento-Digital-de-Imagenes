function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcula costo y gradiente de una red neuronal convolucional simple
% seguida de una capa densa con una funcion objetivo de tipo softmax
%                            
% Parametros:
%  theta      -  unrolled vector de pesos
%  images     -  la lista de imagenes de entrenamiento imageDim x imageDim x numImges
%                
%  numClasses -  numbero de clases
%  filterDim  -  dimensiones del filtro convolucional                            
%  numFilters -  numero de filtros
%  poolDim    -  dimension del area de agrupamiento
%  pred       -  booleano indicando si solo se precisa hacer un fordware
%  propagete y delvolver predicciones
%             
%
%
% Devuelve:
%  cost       -  costro softmax
%  grad       -  gradientes con respecto a theta (si pred==False)
%  preds      -  lista de predicciones para cada ejemplo (si pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width de imagen
numImages = size(images,3); % numbero de imagenes

%% Obtener las matrices de parametros del roll de entrada 
% Wc es la matriz de pesos filterDim x filterDim x numFilters
% bc es el bias correspondiente

% Wd es la matriz densa de la capa escondida con numClasses x hiddenSize
% hiddenSize es el numero de unidades a la salida de la capa de pooling
% bd es su bias correspondient
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% reservar para los gradientes de los parametros.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% PASO 1a: Forward Propagation
%  Este paso propaga la entrada de en la capa convolucional, 
%  sabsampleo (mean pooling), y que luego son usadas como entradas de la 
%  capa de softmax.

%% Convolutional Layer
%  Para cada imagen de entrada y filtro, realizar la convolucion, sumar el
%  bias y aplicar la funcion de activacion sigmoid. Luego, subsamplear
%  con mean pooling. Guardar los resultados de la convolucion en 
%  activations y los resultados del pooling en activationsPooled.
%  Es preciso conservarlos para el backpropagation.
%  Utilizar las funciones cnnConvolve y cnnPool

convDim = imageDim-filterDim+1; % dimension de la salida convolved
outputDim = (convDim)/poolDim; % dimension de la salida subsampled

% tensor convDim x convDim x numFilters x numImages 
activations = zeros(convDim,convDim,numFilters,numImages);
% tensor outputDim x outputDim x numFilters x numImages
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

activations = cnnConvolve(filterDim,numFilters,images,Wc,bc);
activationsPooled = cnnPool(poolDim,activations);

% Redimensionar activations obteniendo 2D matriz, hiddenSize x numImages,
% para Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages); %size 32 x 11

%% Softmax Layer
%  Continuamos con la Forward propagation luego de las pooledActivations 
%  para aplicarlas a una capa softmax standard. probs recibe los resultados
%  numClasses x numImages que va a guardar la probabilidad de la pertenencia
%  de cada imagen a una clase.
probs = zeros(numClasses,numImages);

%%% IMPLEMENTACION AQUI %%%
hiddenSize = size(activationsPooled,1); %size 32

%size(activationsPooled);
%numClasses; %size 10
%hiddenSize;
% Wd --> size numClasses x hiddenSize
% bd -->

for j = 1:numImages
    for k = 1:numClasses
        probs(k,j) = exp(Wd(k,:) * activationsPooled(:,j) + bd(k));
    end
    probs(:,j) = probs(:,j) ./ sum(probs(:,j));
end
%%======================================================================
%% PASO 1b: Calculo del Costo
%  Se van a usar los labels y las probabilidades para calcular la funcion
%  objetivo de tipo softmax. El resultado se guarda en cost.

cost = 0; % inicializo cost

%%% IMPLEMENTAR AQUI %%%


%theta2 = [Wc,bc];
%x = [activationsPooled]
%[Wb, bd] --> 10 x 33

%[M, I] = max(probs);
for m = 1:numImages
    cost = cost + log(probs(labels(m), m));
end
cost = cost / numImages * -1;

%%%%%%%%%%%%%%%%%%%%%%%%%
% Realizar predicciones usando probs y devolver la funcion si calculo de gradientes.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% PASO 1c: Backpropagation
%  Backpropagar los errores a traves de la capa de salida, convolucional y pooling.
%  Guardar los errores para el siguiente paso para el calculo del gradiente.
%  Backpropagar el error para la capa softmax es similar al Ejercicio 1.
%  Para backpropagar en la capa de pooling, hay que des-agrupar el error
%  con respecto a la capa de de pooling para cada filtro y cada imagen.
%  Se puede usar la funcion kron y una matriz de unos para hacerlo de manera
%  eficiente.

%%% IMPLEMENTAR AQUI %%%

% xi --> activationsPooled(:,i)
% activationsPooled --> 32 x 11
% prob --> 10 x 11
% labels --> 11 x 1
% size(Wc) --> 9 x 9 x 2
% size(Wd) --> 10 x 32
% size(bc) --> 2 x 1
% size(bd) --> 10 x 1
% activations --> 20 x 20 x 2 x 11
% images --> 28 x 28 x 11

labelsArr = zeros(numClasses,numImages);
for i = 1:numImages
    labelsArr(labels(i),i) = 1;
end

e = labelsArr - probs;

% size(e) --> 10 x 11

Wd_grad = Wd_grad + e * transpose(activationsPooled) / numImages;
bd_grad = bd_grad + sum(e,2) / numImages;

delta_p2 = transpose(Wd) * e;
delta_p2_upsampled = (1/(poolDim*poolDim))*kron(delta_p2,ones(poolDim));

ddelta = activations .* (1 - activations); 

% size(dsigma) -> 20 x 20 x 2 x 11
% size(sigma_p2_upsampled) --> 160 x 55
delta_c1 = reshape(delta_p2_upsampled, 20, 20, 2, 11) .* ddelta;
for j = 1:numFilters
    for i = 1:numImages 
        Wc_grad(:,:,j) = Wc_grad(:,:,j) + conv2(images(:,:,i),rot90(delta_c1(:,:,j,i)),'valid');
    end
    Wc_grad(:,:,j) = Wc_grad(:,:,j)/numImages;
    bc_grad(j) = sum(sum(sum(delta_c1(:,:,j,:))))/numImages;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%======================================================================
%% PASO 1d: Calculo del Gradiente
%  Luego de propagar los errores, deben ser usados para calcular el gradiente
%  con respecto a los pesos. El calculo del gradiente de la capa softmax se 
%  calculo de la manera usual. Para calcular el gradiente de un filtro en la capa convolucional
%  se debe hacer la convolucion entren el error propagado de ese filtro 
%  para cada imagen y acumularlo sobre todas las imagenes.

%%% YOUR CODE HERE %%%


%% Desenrollar los gradientes en un vector para usar luego por minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
