function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%  cnnPool Pools los descriptores provenientes de la convolucion
%  La funcion usa el Pool promedio
% Parametros:
%  poolDim - dimension de la regiom de pool
%  convolvedFeatures - los descriptores a realizar el pool 
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Devuelve:
%  pooledFeatures - matriz de los features agrupados
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);
pooledSize = convolvedDim / poolDim;

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instrucciones:
%   Realizar el pool de los features en regiones de tamaño poolDim x poolDim,
%   para obtener la matriz pooledFeatures de 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 

%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) es el valor 
%   del descriptor featureNum de la imagen imageNum agrupada sobre la 
%   region (poolRow, poolCol). 
%

%%% IMPLEMENTAR AQUI %%%
for nF = 1:numFilters
    for nI = 1:numImages
        for poolRow = 1:pooledSize
            for poolCol = 1:pooledSize
                sH = (poolRow-1)*poolDim+1;
                eH = (poolRow-1)*poolDim+poolDim;
                sW = (poolCol-1)*poolDim+1;
                eW = (poolCol-1)*poolDim+poolDim;
                promedio = mean2(convolvedFeatures(sH:eH,sW:eW,nF, nI));
                pooledFeatures(poolRow, poolCol, nF, nI) = promedio;
            end
        end
    end
end
%pooledFeatures