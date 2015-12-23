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
    
    # load position and velocity variables from trajectory text file
    x0, y0, z0, vx0, vy0, vz0 = np.loadtxt('/home/cs205/cs205-homework/HW0/P4/P4_trajectory.txt',delimiter=',',unpack=True)
    
    # plot position vectors wrt time on 3D axes
    ax.plot(x0, y0, z0,'--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    
    # create scaling matrix
    R = np.matrix([[1/rx,0,0],[0,1/ry,0],[0,0,1/rz]])

    # load measurements file as rows
    m = np.loadtxt('/home/cs205/cs205-homework/HW0/P4/P4_measurements.txt',delimiter=',')
    
    # for each position given in measurements.txt, 
    # matrix multiply scaling matrix R to position vector, 
    # then typecast to a 1D array
    adj = np.asarray([np.squeeze(np.asarray(np.dot(R,m[i]))) for i in np.arange(0,m.shape[0],1)])
    
    # transpose adj vector to get observed (x, y, z) coordinates
    x_obs, y_obs, z_obs = adj.transpose()
    
    # plot observed positions
    ax.plot(x_obs, y_obs, z_obs,'.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    
    # initialize A matrix and a vector
    A = np.matrix([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1-c*dt,0,0],[0,0,0,0,1-c*dt,0],[0,0,0,0,0,1-c*dt]])
    a = np.array([0,0,0,0,0,g*dt])
    
    # initialize list S with value of s0
    S = [np.array([0,0,2,15,3.5,4.0])]
    
    # for K-1 steps, operate on the last element of S, 
    # convert value to 1D array, and append to end of S
    for k in xrange(1,K-1):
        S.append(np.squeeze(np.asarray(np.dot(A,S[len(S)-1])+a)))
    
    # transpose list of arrays to obtain (x, y, z) coordinates
    x_pred, y_pred, z_pred = np.asarray(S).transpose()[0:3]
    
    ax.plot(x_pred, y_pred, z_pred,'-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    
    # initialize B and C matrices
    B = np.diag(np.array([bx, by, bz, bvx, bvy, bvz]))
    C = np.asmatrix(np.concatenate((np.diag(np.array([rx,ry,rz])),np.diag(np.array([0,0,0]))),axis=1))

    # initialization of s0 and Sigma0
    s0 = np.array([0,0,2,15,3.5,4.0])
    Sigma0 = 0.01 * np.identity(6)
    
    # predictS gives intermediate estimate ~s based on known s_k 
    def predictS(s_k):
        return np.dot(A,s_k)+a #matrix
    
    # predictSig predicts intermediate ~Sigma given Sigmak
    def predictSig(Sig_k):
        return np.linalg.inv(A*Sig_k*A.transpose() + B*B.transpose()) #matrix
        
    # updateSig takes covariance matrix predictor (from predictSig) and outputs (k+1)th Sigma
    def updateSig(Sig_pred):
        return np.linalg.inv(Sig_pred + C.transpose() * C) #matrix
        
    # updateS calculates value of (k+1)th s
    def updateS(Sig_k,s_k,l):
        return updateSig(predictSig(Sig_k)) * (predictSig(Sig_k) * predictS(s_k).T + C.T * np.asmatrix(m[l+1]).transpose()) #matrix

    # initialize list S with value of s0
    S = [s0]
    Sigma_next = Sigma0
        
    # for K-1 steps, operate on the last element of S, 
    # convert value to 1D array, and append to end of S
    for k in xrange(1,K-1):
        S.append(np.squeeze(np.asarray(updateS(Sigma_next,S[len(S)-1],k))))
        Sigma_next = updateSig(predictSig(Sigma_next))
    
    # transpose list of arrays to obtain (x, y, z) coordinates
    x_filt, y_filt, z_filt = np.asarray(S).T[0:3]
        
    ax.plot(x_filt, y_filt, z_filt,'-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
