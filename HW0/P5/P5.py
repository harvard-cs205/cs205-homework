import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    N = np.arange(1, 1001)
    multi = np.ceil(np.log2(N))
    single = N - 1

    # Create axes
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(N, multi,
             '-b', label='Multiple Cashier')
    ax.plot(N, single,
             '-r', label='Single Cashier')

    # Show the plot
    ax.legend()
    plt.show()
