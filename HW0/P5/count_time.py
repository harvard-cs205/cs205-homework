import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

def findRem(N):
    i=1
    while (N>pow(2,i)):
        i =i+1
    return (i-1,N-pow(2,(i-1))) 
    
# def CountTime(N):
#     if (N==1):
#         return 1
#     else:
        
if __name__ == '__main__':
    Nmax=1000
    N=range(2,Nmax)
    t=np.zeros(np.shape(N))
    for i in range(0,Nmax-2):
        t[i],rem=findRem(i)
        if rem>0:
            t[i]=t[i]+1

#    ax=Axes(plt.figure())
    
    plt.loglog(N,range(1,Nmax-1),'-b',label='Not Parallel')
    plt.loglog(N,t, '-r',label='Parallel')
    plt.xlabel('Number of Bags')
    plt.ylabel('Time (sec)')
    plt.title('Time to count assuming no communication time')
    plt.legend(loc=2)
    plt.show()