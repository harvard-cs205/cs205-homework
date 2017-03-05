import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N=np.arange(1,2**7,1)
    x=np.ceil(np.log2(N))
    y=N-1
    
    plt.plot(N,x,".b",label="infinite cashiers")
    plt.plot(N,y,".r",label="one cashier")
    plt.legend(loc="upper left")
    plt.title("The relationship between the number of bags and total time spent")
    plt.xlabel("the number of bags")
    plt.ylabel("total time spent")
    plt.show()
