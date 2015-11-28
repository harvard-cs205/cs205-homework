import matplotlib.pyplot as plt
import numpy as np
import math

# Main
if __name__ == '__main__':

    fig = plt.figure()
    ax = plt.subplot(111)

    t = np.logspace(1, 5, num=5, base=2)

    ax.plot(t, t-1, '-og', label='alone')
    ax.plot(t, np.log2(t), '-or', label='infinite')
    ax.legend()

    plt.xlabel('# of bags')
    plt.ylabel('time')
    plt.title('Time needed to count bags')
    plt.show()
