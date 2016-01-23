import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

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

    data1 = np.loadtxt('P4_trajectory.txt', delimiter=",")
    x_coords = data1[:, 0]
    y_coords = data1[:, 1]
    z_coords = data1[:, 2]


    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    data2 = np.loadtxt('P4_measurements.txt', delimiter=",")
    C = np.matrix([[1.0/rx, 0, 0], [0, 1.0/ry, 0], [0, 0, 1.0/rz]])
    #print C
    result = np.dot(data2, C)
    x_coords = np.asarray(result[:, 0]).reshape(-1)
    y_coords = np.asarray(result[:, 1]).reshape(-1)
    z_coords = np.asarray(result[:, 2]).reshape(-1)
    # print z_coords


    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1-c*dt, 0,0],[0,0,0,0, 1-c*dt,0],[0,0,0,0,0,1-c*dt]])
    a = np.matrix([0,0,0,0,0,g*dt]).T
    s = np.matrix([0,0,2,15,3.5,4.0]).T

    preds = np.zeros([6, K])

    for i in xrange(K):
        s_i = np.dot(A, s) + a
        for j in xrange(len(s_i)):
            preds[j, i] = s_i[j]
        s = s_i


    x_coords = preds[0]
    y_coords = preds[1]
    z_coords = preds[2]

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.matrix([[bx,0,0,0,0,0], [0,by,0,0,0,0], [0,0,bz,0,0,0], [0,0,0,bvx,0,0], [0,0,0,0,bvy,0], [0,0,0,0,0,bvz]])
    C = np.matrix([[rx, 0, 0,0,0,0], [0, ry, 0,0,0,0], [0, 0, rz,0,0,0]])

    measures = np.loadtxt('P4_measurements.txt', delimiter=",")

    def predictS(A, s_k, a):
        # print A.shape
        # print s_k.shape
        # print a.shape
        return np.dot(A, s_k) + a

    def predictSig(Sig_k, A, B):
        return np.linalg.inv(np.dot(np.dot(A, Sig_k), A.T) + np.dot(B, B.T))

    def updateSig(Sig, C):
        return np.linalg.inv(Sig + np.dot(C.T, C))

    def updateS(Sig_k1, Sig, s, C, m):
        a = (np.dot(Sig, s) + np.dot(C.T, m))
        b = Sig_k1
        return np.dot(b, a)
        #return np.dot(Sig_k1, (np.dot(Sig, s) + np.dot(C.T, m)))


    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    s0 = np.matrix([0,0,2,15,3.5,4.0]).T
    sig0 = 0.01*np.identity(6)

    filt = np.zeros([6, K])

    for k in xrange(K):
        s = predictS(A, s0, a)
        Sig = predictSig(sig0, A, B)
        Sig_k1 = updateSig(Sig, C)
        m = measures[k].reshape((1,3))
        s_new = updateS(Sig_k1, Sig, s, C, m.T)
        for j in xrange(len(s_new)):
            filt[j, k] = s_new[j]
        print s_new
        s0 = s_new
        sig0 = Sig_k1


    x_coords = filt[0]
    y_coords = filt[1]
    z_coords = filt[2]

    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    ax.legend()
    plt.show()

    # Show the plot




















