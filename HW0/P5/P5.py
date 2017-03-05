import numpy as np
import math
import matplotlib.pyplot as plt

def predictS(A,s,a):
  return A*s+a

def predictSig(A,Sigma,B):
  return np.linalg.inv(A*Sigma*A.T + B*B.T)

def updateSig(Sigma,C):
  return np.linalg.inv(Sigma+C.T*C)

def updateS(Sigma_next,Sigma_appr,s,C,m):
  return Sigma_next*(Sigma_appr*s+C.T*m)

if __name__ == '__main__':
    # P5.4
    N = [float(2**exp) for exp in range(1,11)]
    print N
    Time_for_inf = [math.log(x,2) for x in N]
    Time_for_alone = [x-1 for x in N]
 
    plt.plot(N,Time_for_inf, "-b")
    plt.plot(N,Time_for_alone, "--r")
    plt.xlabel("Num of Bags")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()
