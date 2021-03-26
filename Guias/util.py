def calcNivelesGrises(img):
    L = 0
    for x in range(len(img)):
        for y in range(len(img[0])):
            L = max(L, img[x][y]) 
    return L+1
