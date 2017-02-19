import numpy as np
from pylab import *

if __name__ == '__main__':
    N = 65

    x_coords = range(N)
    y_coords = [x - 1 for x in x_coords]
    y_coords[0] = 0

    xlabel('number of bags')
    ylabel('time in seconds')
    plot(x_coords, y_coords, 
            '--b', label='Lone Cashier')

    y_coords = [np.ceil(np.log2(x)) for x in x_coords]
    y_coords[0] = 0

    plot(x_coords, y_coords, 
            '--r', label='Infinite Cashiers')
    # Show the plot
    legend()
    savefig('P5.png')
    show()