import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    numpoints = 300
    N = range(1, numpoints + 1)
    inf = np.ceil(np.log2(N))
    lone = range(0, numpoints);

    plt.plot(N, inf, '-b', label='infinite employees')
    plt.plot(N, lone, '--g', label='lone cashier')
    plt.xlabel('Number of bags')
    plt.ylabel('Counting time in seconds')
    plt.legend()
    plt.show()