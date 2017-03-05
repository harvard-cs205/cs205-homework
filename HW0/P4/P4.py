import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    # Model parameters
    K = 120    # Number of times steps
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
    C = np.zeros((3,3))
    C[0,0] = rx
    C[1,1] = ry
    C[2,2] = rz

    # Create 3D axes for plotting
    ax = Axes3D(plt.figure())

    #####################
    # Part 1:
    #
    # Load true trajectory and plot it
    # Normally, this data wouldn't be available in the real world
    #####################
    s_true = np.loadtxt('P4_trajectory.txt',delimiter=',')
    x_true = s_true[:,0]
    y_true = s_true[:,1]
    z_true = s_true[:,2]
    
    ax.plot(x_true, y_true, z_true,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    s_obs = np.loadtxt('P4_measurements.txt',delimiter=',')
    s_obs_pos = s_obs[:,0:3]
    s_obs_pos_adjust = np.transpose(np.dot(np.linalg.inv(C),np.transpose(s_obs_pos)))
    x_obs_pos_adjust = s_obs_pos_adjust[:,0]
    y_obs_pos_adjust = s_obs_pos_adjust[:,1]
    z_obs_pos_adjust = s_obs_pos_adjust[:,2]

    ax.plot(x_obs_pos_adjust, y_obs_pos_adjust, z_obs_pos_adjust,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.zeros((6,6))
    A[0,np.array([0,3])] = np.array([1,dt])
    A[1,np.array([1,4])] = np.array([1,dt])
    A[2,np.array([2,5])] = np.array([1,dt])
    A[3,3] = 1 - c*dt
    A[4,4] = 1 - c*dt
    A[5,5] = 1 - c*dt  
    a = np.zeros(6)
    a[5] = g*dt
    s = np.zeros((6,K))

    # Initial conditions for s0
    s[np.array([2,3,4,5]),0] = np.array([2,15,3.5,4])
    # Compute the rest of sk using Eq (1)
    for ii in range(1,K):
    	s[:,ii] = np.dot(A,s[:,ii-1]) + a

    ax.plot(s[0,:], s[1,:], s[2,:],
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.zeros((6,6))
    B[0,0] = bx
    B[1,1] = by
    B[2,2] = bz
    B[3,3] = bvx
    B[4,4] = bvy
    B[5,5] = bvz

    # Initial conditions for s0 and Sigma0
    sFiltered = np.zeros((6,K))
    sFiltered[np.array([2,3,4,5]),0] = np.array([2,15,3.5,4])
    sigmaCurrent = 0.01*np.identity(6)
    C = np.zeros((3,6))
    C[0,0] = rx
    C[1,1] = ry
    C[2,2] = rz
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    for ii in range(1,K):
    	sPred = np.dot(A,sFiltered[:,ii-1]) + a
    	sigTilde = np.linalg.inv(np.asmatrix(A)*np.asmatrix(sigmaCurrent)*np.asmatrix(np.transpose(A)) + np.dot(B,np.transpose(B)))
	sigmaCurrent = np.linalg.inv(sigTilde + np.dot(np.transpose(C),C))
	firstPart = np.reshape(np.dot(sigTilde,sPred),(6,1))
	secondPart = np.reshape(np.dot(np.transpose(C),np.transpose(s_obs[ii,:])),(6,1))
    	sFiltered[:,ii] = np.reshape(np.dot(sigmaCurrent,(firstPart+secondPart)),6)

    ax.plot(sFiltered[0,:], sFiltered[1,:], sFiltered[2,:],
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
