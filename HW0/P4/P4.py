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
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',')
    x_coords = s_true[:,0]
    y_coords = s_true[:,1]
    z_coords = s_true[:,2]
    #####################

    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    s_measure = np.loadtxt('P4_measurements.txt', delimiter=',')
    
    V = np.matrix([[1/rx, 0, 0], [0, 1/ry, 0], [0, 0, 1/rz]])
    
    x_coordsb = []
    y_coordsb = []
    z_coordsb = []

    for row in s_measure:
    	arr = np.transpose([row])
        n = V*arr
    	x_coordsb.append(float(n[0]))
    	y_coordsb.append(float(n[1]))
    	z_coordsb.append(float(n[2]))

    ax.plot(x_coordsb, y_coordsb, z_coordsb,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix([[1, 0, 0, dt, 0, 0], \
    	[0, 1, 0, 0, dt, 0], \
    	[0, 0, 1, 0, 0, dt], \
    	[0, 0, 0, 1 - c*dt, 0, 0], \
    	[0, 0, 0, 0, 1 - c*dt, 0], \
    	[0, 0, 0, 0, 0, 1 - c*dt]])

    a = np.transpose(np.matrix([0, 0, 0, 0, 0, g*dt]))

    s = np.transpose(np.matrix([0, 0, 2, 15, 3.5, 4.0]))
    
    x_coordsc = []
    y_coordsc = []
    z_coordsc = []

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    for i in range(K-1):	
    	x_coordsc.append(float(s[0]))
    	y_coordsc.append(float(s[1]))
    	z_coordsc.append(float(s[2]))
    	s = A*s+a

    ax.plot(x_coordsc, y_coordsc, z_coordsc,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.matrix([[bx, 0, 0, 0, 0, 0], \
    	[0, by, 0, 0, 0, 0], \
    	[0, 0, bz, 0, 0, 0], \
    	[0, 0, 0, bvx, 0, 0], \
    	[0, 0, 0, 0, bvy, 0],\
    	[0, 0, 0, 0, 0, bvz]])

    C = np.matrix([[rx, 0, 0, 0, 0, 0], \
    	[0, ry, 0, 0, 0, 0], \
    	[0, 0, rz, 0, 0, 0]])

    # Initial conditions for s0 and Sigma
    sigma = np.identity(6)*0.01
    s = np.transpose(np.matrix([0, 0, 2.0, 15.0, 3.5, 4.0]))

    S = np.matrix([0, 0, 2.0, 15.0, 3.5, 4.0])

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    def predictS(s):
        s_approx = A*s+a
        return s_approx

    def predictSig(sigma):
        sigma_approx = np.linalg.matrix_power(A*sigma*np.transpose(A)+B*np.transpose(B), -1)
        return sigma_approx

    def updateSig(sigma_approx):
        sigma = np.linalg.matrix_power(sigma_approx + np.transpose(C)*C, -1)
        return sigma

    def updateS(s_approx, sigma_approx, sigma):
        s = sigma*(sigma_approx*s_approx+np.transpose(C)*np.transpose(np.matrix(s_measure[i+1])))
        return s

    x_coordsd = []
    y_coordsd = []
    z_coordsd = []

    for i in range(120):
        s_approx = predictS(s)
        sigma_approx = predictSig(sigma)
        sigma = updateSig(sigma_approx)
        s = updateS(s_approx, sigma_approx, sigma)
        x_coordsd.append(float(s[0]))
        y_coordsd.append(float(s[1]))
        z_coordsd.append(float(s[2]))

    ax.plot(x_coordsd, y_coordsd, z_coordsd,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
