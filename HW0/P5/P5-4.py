import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # number of bags
    x = np.arange(2,501)
    # time needed to count for infinite cashiers
    y1 = np.ceil(np.log2(x))
    # time needed to count for 1 cashier
    y2 = x - 1

    # Show the plot
    plt.plot(x, y1, '-b', label='Infinite Cashiers')
    plt.plot(x, y2, '--r', label='One Cashier')
    plt.xlabel('No. of Bags')
    plt.ylabel('Counting Time')
    plt.legend(loc='upper left')
    plt.savefig('P5.png')
    plt.show()

