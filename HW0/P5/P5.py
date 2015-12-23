import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.arange(1,500)
    y1 = x - 1
    y2 = np.ceil(np.log2(x))
    
    plt.plot(x,y1,'-b')
    plt.plot(x,y2,'-g')
    plt.yscale('log')
    plt.xlabel('number of bags')
    plt.ylabel('time (log seconds)')
    plt.title('bags processed per second')
    plt.show()
