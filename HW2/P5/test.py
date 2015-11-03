import numpy as np

def weight_gen(grid):
    weight = np.zeros_like(grid).astype(int)
    x,y,idx = 0,0,1
    while x<(int)(weight.shape[0]/2)*2 and y<(int)(weight.shape[1]/2)*2:
        for j in range(x+1):
            weight[x,j] = idx
            idx += 1
        for i in reversed(range(y)):
            weight[i,y] = idx
            idx += 1
        for i in range(y+2):
            weight[i,y+1] = idx
            idx += 1
        for j in reversed(range(x+1)):
            weight[x+1,j] = idx
            idx += 1
        x = x+2
        y = y+2
    return weight


print weight_gen(np.zeros((6,6)))