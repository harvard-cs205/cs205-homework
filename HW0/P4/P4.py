import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    # Model parameters
    K = 121    # Number of times steps
    dt = 0.01  # Delta t
    c = 0.1    # Coefficient of drag
    g = -9.81  # Gravity
    # For constructing matrix B
    bx = 0
    by = 0
    bz = 0
    bvx = 0.25
    bvy = 0.25
    bvz = 0.1
    # For constructing matrix C
    rx = 1.0
    ry = 5.0
    rz = 5.0

    # Create 3D axes for plotting
    ax = Axes3D(plt.figure())

    #####################
    # Part 1:
    #
    # Load true trajectory and plot it
    # Normally, this data wouldn't be available in the real world
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',', dtype={'names':('x', 'y', 'z'), 'formats': (np.float, np.float, np.float)}, usecols = (0,1,2))
    #####################
    #s_true = np.loadtxt('P4_trajectory.txt', delimiter=',', usecols = (0,1,2))
    # ax.plot(x_coords, y_coords, z_coords,
    #         '--b', label='True trajectory
    arrX = s_true['x']
    arrY = s_true['y']
    arrZ = s_true['z']
  
    for s in s_true:
        ax.plot(arrX, arrY, arrZ, '--b', label='True trajectory')
        
    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
        
    m = np.loadtxt('P4_measurements.txt', delimiter=',', dtype={'names':('mx', 'my', 'mz'), 'formats': (np.float, np.float, np.float)}, usecols = (0,1,2))
    #####################

    # ax.plot(x_coords, y_coords, z_coords,
    #         '.g', label='Observed trajectory')
    
    
    arrMX = 1. / rx * m['mx']
    arrMY = 1. / ry * m['my']
    arrMZ = 1. / rz * m['mz']
    
    ax.plot(arrMX, arrMY, arrMZ, '.g', label='Observed trajectory')
    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
