__author__ = 'Haosu Tang'

import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    N = range(1,2000)
    plt.plot(N, [int(math.log(x, 2)+0.5) for x in N])
    plt.show()
