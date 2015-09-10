import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys


# Define functions here


def predictS(A,s,a):
	s_tilde = A.dot(s) + a
	return s_tilde

def predictSig(Sig, A, B):
	Sig_tilde = (A.dot(Sig)).dot(A.T) + B.dot(B.T)
	Sig_tilde = np.linalg.inv(Sig_tilde)
	return Sig_tilde

def updateSig(Sig_tilde, C):
	Sig_update = Sig_tilde + (C.T).dot(C)
	sig_update = np.linalg.inv(Sig_update)
	return Sig_update

def updateS(Sig_update, Sig_tilde, s_tilde, C, m):
	s_update = Sig_update.dot(Sig_tilde.dot(s_tilde) + (C.T).dot(m)  )
	return s_update


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


    #####################
    # Part 1:
    #
    # Load true trajectory and plot it
    # Normally, this data wouldn't be available in the real world
    #####################

    # Read the data
    s_true = pd.read_csv('P4_trajectory.txt',header= None, names= ['x','y','z','vx','vy','vz'])
    s_true.head()
    x_coords = s_true.x
    y_coords = s_true.y
    z_coords = s_true.z
    plt.close("all")
    ax = Axes3D(plt.figure())
    ax.plot(x_coords, y_coords, z_coords,
        '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    s_obs = pd.read_csv('P4_measurements.txt',header= None, names= ['x','y','z'])
    s_obs.head()

    ax.plot(s_obs.x/rx, s_obs.y/ry, s_obs.z/rz,'.g', label='Observed trajectory');


    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    A = np.array([[1,0,0,dt,0,0],
        [0,1,0,0,dt,0],
        [0,0,1,0,0,dt],
        [0,0,0,1-c*dt,0,0]
        ,[0,0,0,0,1-c*dt,0]
        ,[0,0,0,0,0,1-c*dt]])
    a = np.zeros([6,1])
    a[-1,0] = g*dt
    s = np.array([[0],[0],[2],[15],[3.5],[4.0]])
    print s_true.shape
    # Create matrix to store the s's
    s_blind = np.zeros((6,K-1))
    s_blind[:,0] = s[:,0]  
    print s_blind[:,K-2]

    for k in range(1,K-1):
        s = A.dot(s) + a
        s_blind[:,k] = s[:,0]
    
    # Now extract values of x, y and z
    x_blind = s_blind[0,:]
    y_blind = s_blind[1,:]
    z_blind = s_blind[2,:]
    ax.plot(x_blind, y_blind, z_blind,'-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    ####################


    # B = ?
    # C = ?

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
