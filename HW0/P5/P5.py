import numpy as np
import math
import matplotlib.pyplot as plt

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
    plt.savefig('P5.png')
    plt.show()
