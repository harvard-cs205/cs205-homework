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
    s_true = np.loadtxt('P4_trajectory.txt', delimiter = ',')
    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2],'--b', label = 'True trajectory')

    #####################
    # Part 2:
    # Read the observation array and plot it (Part 2)
    s_measure = np.loadtxt('P4_measurements.txt', delimiter= ',')
    r_matrix = np.matrix([[1/rx,0,0], [0,1/ry,0], [0,0,1/rz]])
    s_obs = s_measure*r_matrix
    s_obs = np.squeeze(np.asarray(s_obs))
    ax.plot(s_obs[:,0], s_obs[:,1], s_obs[:,2],'.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    A = np.matrix([[1,0,0,dt,0,0], [0,1,0,0,dt,0], [0,0,1,0,0,dt], 
    [0,0,0,1-(c*dt),0,0], [0,0,0,0,1-(c*dt),0], [0,0,0,0,0,1-(c*dt)]])
    a = np.matrix([[0,0,0,0,0,g*dt]]).T
    # Initial conditions for s0
    s0 = np.matrix([[0,0,2,15,3.5,4.0]]).T
    # Compute the rest of sk using Eq (1)
    s_simple = np.zeros(shape = (6,K))
    s_simple[:,0] = np.squeeze(np.asarray(s0))
    for i in range(K-1):
        s_simple[:,i+1]=np.squeeze(np.asarray(A*np.matrix(s_simple[:,i]).T + a))
    # Generate the plot
    ax.plot(s_simple[0,:], s_simple[1,:], s_simple[2,:],'-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    def predictS(sk):
        return (A*sk+a)
    def predictSig(sigk):
        return ((A*sigk*A.T + B*B.T).I)
    def updateSig(sig):
        return ((sig + C.T*C).I)
    def updateS(sigk1,mk1,sig,s):
        return (sigk1*(sig*s + C.T*mk1))
        
    B = np.matrix([[bx,0,0,0,0,0], [0,by,0,0,0,0], [0,0,bz,0,0,0], 
    [0,0,0,bvx,0,0], [0,0,0,0,bvy,0], [0,0,0,0,0,bvz]])
    C = np.matrix([[rx,0,0,0,0,0], [0,ry,0,0,0,0], [0,0,rz,0,0,0]])

    # Initial conditions for s0 and Sigma0
    s0=s0
    sigma0=0.001*np.identity(6)
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    s_kal = np.zeros(shape = (6,K))
    s_kal[:,0] = np.squeeze(np.asarray(s0))
    sigk = sigma0
    for i in range(K-1):
        s = predictS(sk=np.matrix(s_kal[:,i]).T)
        sig = predictSig(sigk=sigk)
        sigk = updateSig(sig=sig)
        s_kal[:,i+1]=np.squeeze(np.asarray(updateS(sigk1=sigk,sig=sig,s=s,
        mk1=np.matrix(s_measure[i+1,:]).T)))
    ax.plot(s_kal[0,:], s_kal[1,:], s_kal[2,:],'-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
