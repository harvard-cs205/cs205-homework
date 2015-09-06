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
    #####################
    s_true = np.loadtxt('P4_trajectory.txt', delimiter = ',', dtype = [('x_coords',float),('y_coords',float),('z_coords',float),('vx',float),('vy',float),('vz',float)])
    x_coords =  s_true['x_coords']
    y_coords =  s_true['y_coords']
    z_coords =  s_true['z_coords']

    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    ##################### 
    s_obs = np.loadtxt('P4_measurements.txt',delimiter =',')
    C = np.matrix([[1/rx,0,0],[0,1/ry,0],[0,0,1/rz]])
    meas = s_obs * C;
    ax.plot(s_obs[:,0],s_obs[:,1],s_obs[:,2],
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    A = np.concatenate((np.concatenate((np.eye(3),dt*np.eye(3))\
        ,axis=1),np.concatenate((np.zeros((3,3)),\
	(1-c*dt)*np.eye(3)),axis=1)),axis = 0)  
    a = np.matrix([0,0,0,0,0,g*dt])
    a = a.T 
    s = np.matrix(np.zeros((6,K)))
    

    # Initial conditions for s0
    s[:,0] = np.matrix([0,0,2,15,3.5,4.0]).T
    # Compute the rest of sk, using Eq (1)
    for i in range(1,K):
        s[:,i] = A*np.matrix(s[:,i-1]) + a
    s = np.array(s)
    ax.plot(s[0,:], s[1,:],s[2,:],\
         '-k', label='Blind trajectory')

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
