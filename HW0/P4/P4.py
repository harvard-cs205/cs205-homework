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

    s_true = np.loadtxt('/home/jocelyn/cs205-homework/HW0/P4/P4_trajectory.txt', delimiter = ',')

    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2],
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    obs = np.loadtxt('/home/jocelyn/cs205-homework/HW0/P4/P4_measurements.txt', delimiter = ',')
    C = np.array([[1.0/rx, 0,0], [0,1.0/ry, 0],[0,0,1.0/rz]])
   
    traj = np.dot(C, np.transpose(obs))

    ax.plot(traj[0,:], traj[1,:], traj[2,:],
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix([[1.,0,0,dt,0,0],[0,1.,0,0,dt,0,],[0,0,1.,0,0,dt],[0,0,0,1-c*dt,0,0],[0,0,0,0,1-c*dt,0],[0,0,0,0,0,1-c*dt]])
    a = np.matrix([[0,0,0,0,0,g*dt]]).T
    # Initial conditions for s0
    s = np.zeros([6,K])
    s = np.asmatrix(s)
    s[:,0] = np.matrix([[0,0,2,15,3.5,4]]).T

    # Compute the rest of sk using Eq (1)
    for i in xrange(1,K):
        s[:,i] = A*s[:,i-1] + a
    s = np.squeeze(np.asarray(s))
    ax.plot(s[0,:], s[1,:], s[2,:],
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction

    # I CAN'T SEEM TO FIND THE PROBLEM BUT MY CURVE DEFINITELY GOES THE WRONG WAY.

    #####################
     
    # Initial conditions for s0 and Sigma0   
    B = np.zeros([6,6])
    B[3,3] = bvx
    B[4,4] = bvy
    B[5,5] = bvz
    B = np.asmatrix(B)
    # Add columns to C from part 1
    C = np.array([[1.0/rx, 0,0,0,0,0], [0,1.0/ry, 0,0,0,0],[0,0,1.0/rz,0,0,0]])
    C = np.asmatrix(C)
    #Observation Data
    obs = np.asmatrix(obs)

    def predictS(A, s_init, a):
        pred_s = A*s_init + a        
        return pred_s

    def predictSig(A, sig_init, B):
        pred_sig = (A*sig_init*A.T + B*B.T).I
        return pred_sig

    def updateSig(pred_sig, C):
        new_sig = (pred_sig + C.T*C).I
        return new_sig

    def updateS(pred_s, pred_sig, new_sig, C, measured):
        new_s = new_sig*(pred_sig*pred_s + C.T*measured)  
        return new_s   
    
    sk = np.zeros([6,K])
    sk = np.asmatrix(sk)
    sk[:,0] = np.matrix([[0,0,2,15,3.5,4]]).T 
    sigma0 = 0.01*np.matrix(np.identity(6))
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
        
    pred_s = predictS(A, sk[:,0], a)
    pred_sig = predictSig(A, sigma0, B)
    new_sig = updateSig(pred_sig, C)
    sk[:,1] = updateS(pred_s, pred_sig, new_sig, C, obs.T[:,1])

    for j in xrange(1,K-1):
        pred_s = predictS(A, sk[:,j], a)
        pred_sig = predictSig(A, new_sig, B)
        new_sig = updateSig(pred_sig, C)
        sk[:,j+1] = updateS(pred_s, pred_sig, new_sig, C, obs.T[:,j+1])
    
    sk_array = np.squeeze(np.asarray(sk))

    ax.plot(sk_array[0,:], sk_array[1,:], sk_array[2,:],
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()



