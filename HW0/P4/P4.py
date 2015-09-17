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
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',')
    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2],
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    m = np.loadtxt('P4_measurements.txt', delimiter=',')
    s_obs = m * np.array([1./rx, 1./ry, 1./rz]) 
    ax.plot(s_obs[:,0], s_obs[:,1], s_obs[:,2],
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Construct matrix A from blocks
    sub_a = np.eye(3)
    sub_b = np.eye(3) * dt
    sub_c = np.zeros((3,3))
    sub_d = np.eye(3) * (1-c*dt)
    A = np.bmat([[sub_a, sub_b], [sub_c, sub_d]])
    A = np.array(A)
    a = np.array([0.]*5 + [g*dt])

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    # Initialize s0
    s0 = s_true[0,:]
    S_pred0 = np.zeros((6,K))
    S_pred0[:,0] = s0
    s = s0
    for i in range(1,K):
        s = np.dot(A,s) + a
        S_pred0[:,i] = s

    ax.plot(S_pred0[0,:], S_pred0[1,:], S_pred0[2,:],
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    def predictS(A, s, a):
        '''Use Eq(2) to compute intermediate s'''
        return np.dot(A,s) + a

    def predictSig(A, B, Sig):
        '''
        Use Eq(3) to compute cov mat.
        Returns an array.
        '''
        # Make sure inputs are matrices
        A = np.asmatrix(A)
        B = np.asmatrix(B)
        Sig = np.asmatrix(Sig)
        return np.asarray((A*Sig*A.T + B*B.T).I)

    def updateSig(SigInt, C):
        '''
        Use Eq(4) to update cov mat.
        Returns an array
        '''
        # Make sure inputs are matrices
        SigInt = np.asmatrix(SigInt)
        C = np.asmatrix(C)
        return np.asarray((SigInt + C.T*C).I)

    def updateS(SigNext, SigInt, C, s, m):
        '''
        Use Eq(5) to update s.
        Returns an array
        '''
        return np.dot(SigNext, np.dot(SigInt,s)+np.dot(C.T,m))

    B = np.diag(np.array([bx,by,bz,bvx,bvy,bvz]))
    C = np.hstack((np.diag(np.array([rx,ry,rz])), np.zeros((3,3))))

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    
    # Initialize s0 and Sig0
    s = s_true[0,:]
    Sig = np.eye(6)*0.001
    S_pred1 = np.zeros((6,K))
    S_pred1[:,0] = s
    for i in range(1,K): 
        s = predictS(A, s, a)
        SigInt = predictSig(A, B, Sig)
        Sig = updateSig(SigInt, C)
        s = updateS(Sig, SigInt, C, s, m[i])
        S_pred1[:,i] = s

    ax.plot(S_pred1[0,:], S_pred1[1,:], S_pred1[2,:],
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
