import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys



# Define the functions
def predictS(A,s,a):
	s_tilde = A.dot(s) + a
	return s_tilde

def predictSig(Sig, A, B):
	Sig_tilde = (A.dot(Sig)).dot(A.T) + B.dot(B.T)
	Sig_tilde = np.linalg.inv(Sig_tilde)
	return Sig_tilde

def updateSig(Sig_tilde, C):
	Sig_update = np.linalg.inv(Sig_tilde + (C.T).dot(C))
	return Sig_update

def updateS(Sig_update, Sig_tilde, s_tilde, C, m):
	s_update = Sig_update.dot(Sig_tilde.dot(s_tilde) + (C.T).dot(m) )
	return s_update

def get_mk(s_obs,k):
    mk = np.zeros((3,1))
    mk[0,0] = s_obs.x[k]
    mk[1,0] = s_obs.y[k]
    mk[2,0] = s_obs.z[k]
    return mk


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
    C = np.zeros((3,6))
    C[0,0] = rx
    C[1,1] = ry
    C[2,2] = rz
    a = np.zeros([6,1])
    a[-1,0] = g*dt
    s = np.array([[0],[0],[2],[15],[3.5],[4.0]])
    # Create matrix to store the s's
    s_blind = np.zeros((6,K-1))
    s_blind[:,0] = s[:,0]  

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

	# Initialization
    s = np.array([[0],[0],[2],[15],[3.5],[4.0]])
    Sig = 0.01*np.eye(6)
    B_diag = np.array([bx,by,bz,bvx,bvy,bvz])
    B = np.diag(B_diag)
    # A is the same as before

    # Create matrix to store the s's
    s_filter = np.zeros((6,K-1))
    s_filter[:,0] = s[:,0]     	    	
  	
    # Write the loop
    for k in range(1,K-1):

        # Compute Sig_tilde eq (3)
        Sig_tilde = predictSig(Sig,A,B)    	  

        # Compute s_tilde eq (2)
        s_tilde = predictS(A,s,a)    	  
          
        # Compute update of Sig
        Sig = updateSig(Sig_tilde, C)

        # Compute udpate of s
        mk = get_mk(s_obs,k)
        s = updateS(Sig, Sig_tilde, s_tilde, C, mk)    	    
        # Store the value
        s_filter[:,k] = s[:,0]    	
    # Now extract values of x, y and z
    x_filter = s_filter[0,:]
    y_filter = s_filter[1,:]
    z_filter = s_filter[2,:]

    ax.plot(x_filter, y_filter, z_filter,'-r', label='Filtered trajectory')
    # Show the plot
    ax.legend()
    plt.show()
