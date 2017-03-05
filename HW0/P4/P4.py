import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from StringIO import StringIO

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

    s_true = np.loadtxt('P4_trajectory.txt', 
             delimiter = ',', unpack = True)
    ax.plot(s_true[0], s_true[1], s_true[2],
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    C_partial = np.matrix([
                            [1.0/rx,0,0],
                            [0,1.0/ry,0],
                            [0,0,1.0/rz]
                         ])
    s_measured = np.loadtxt('P4_measurements.txt', 
             delimiter = ',', unpack = True)

    s_measured = np.matrix(s_measured)
    (cols, rows) = s_measured.shape

    measured_output = np.zeros([cols,rows])
    measured_output = np.asmatrix(measured_output)

    for i in range(0,rows):
        measured_output[:,i] = C_partial * s_measured[:,i]

    measured_output = np.array(measured_output)

    ax.plot(measured_output[0], measured_output[1], measured_output[2],
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    drag_per_dt = 1 - c*dt
    A = np.matrix([
                    [1,0,0,dt,0,0],
                    [0,1,0,0,dt,0],
                    [0,0,1,0,0,dt],
                    [0,0,0,drag_per_dt,0,0],
                    [0,0,0,0,drag_per_dt,0],
                    [0,0,0,0,0,drag_per_dt]
                 ])
    #print A
    a = np.matrix([
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [g*dt]
                 ])
    #print a

    # Initial conditions for s0
    s_0 = np.matrix([
                    [s_true[0][0]],
                    [s_true[1][0]], 
                    [s_true[2][0]], 
                    [s_true[3][0]],
                    [s_true[4][0]],
                    [s_true[5][0]]
                 ])
    #print s_column

    s = np.zeros([6,K])
    s = np.asmatrix(s)

    # construct s using Eq (1)
    s_column = s_0
    s[:,0] = s_column
    for k in range(1,K):
        s[:,k] = A*s_column + a
        s_column = s[:,k]

    s = np.array(s)
    ax.plot(s[0], s[1], s[2],
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.matrix([
                    [bx,0,0,0,0,0],
                    [0,by,0,0,0,0],
                    [0,0,bz,0,0,0],
                    [0,0,0,bvx,0,0],
                    [0,0,0,0,bvy,0],
                    [0,0,0,0,0,bvz]
                 ])
    C = np.matrix([
                    [rx,0,0,0,0,0],
                    [0,ry,0,0,0,0],
                    [0,0,rz,0,0,0],
                 ])

    def predictS (A,s_k,a):
        return A*s_k + a

    def predictSig(A,sig_k,B):
        return np.linalg.inv(A*sig_k*A.transpose() + B*B.transpose())

    def updateSig(sig_tilda, C):
        return np.linalg.inv(sig_tilda + C.transpose()*C)

    def updateS(sig_k1, sig_tilda, s_tilda, C, m_k1):
        return sig_k1 * (sig_tilda*s_tilda + C.transpose()*m_k1)
    
    # Initial conditions for s0 and Sigma0
    sig_0 = 0.01 * np.identity(6)
    m = s_measured

    s = np.zeros([6,K])
    s = np.asmatrix(s)

    # construct s using Eq (1)
    s_k = s_0
    sig_k = sig_0
    
    s[:,0] = s_k
    for k in range(1,K):
        s_tilda = predictS(A,s_k,a)
        sig_tilda = predictSig(A, sig_k, B)
        sig_k1 = updateSig(sig_tilda, C)
        s[:,k] = updateS(sig_k1, sig_tilda, s_tilda, C, m[:,k])

        s_k = s[:,k]
        sig_k = sig_k1

    s = np.array(s)
    ax.plot(s[0], s[1], s[2],
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
