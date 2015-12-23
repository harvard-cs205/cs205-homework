import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print "program started"


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

    #loading files
    s_true = np.loadtxt("P4_trajectory.txt", delimiter = ",")

    x_coords = s_true[:,0]
    y_coords = s_true[:,1]
    z_coords = s_true[:,2]

    ax.plot(x_coords, y_coords, z_coords,
        '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    #loading files
    robot_mk = np.loadtxt("P4_measurements.txt", delimiter = ",")

    # make sure it is a matrix and not array
    robot_mk = np.matrix(robot_mk)

    #make matrix to multiply by to fix stretching
    R = np.matrix([[1/rx, 0, 0], [0, 1/ry, 0], [0, 0, 1/rz]])

    #matrix mult and transpose to get k x 3 matrix -> array
    approx_Xk = R * robot_mk.T
    approx_Xk = approx_Xk.T
    approx_Xk = np.array(approx_Xk)

    x_coords = approx_Xk[:,0]
    y_coords = approx_Xk[:,1]
    z_coords = approx_Xk[:,2]

    ax.plot(x_coords, y_coords, z_coords,
        '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?

    #make a 6 x K matrix to store
    S = np.zeros([6,K])
    S = np.matrix(S)

    # make A
    A = np.zeros([6,6])
    A = np.matrix(A)
    A[0,0] = 1
    A[1,1] = 1
    A[2,2] = 1
    A[0,3] = dt
    A[3,3] = 1 - (c*dt)
    A[1,4] = dt
    A[4,4] = 1 - (c*dt)
    A[2,5] = dt
    A[5,5] = 1 - (c*dt)

    # make a
    a = np.matrix([0, 0, 0, 0, 0, g*dt]).T

    # Initial conditions for s0
    s = np.matrix([0, 0, 2, 15, 3.5, 4.0]).T
    S[:,0] = s

    # and propogate
    for i in range(K-1):
        s = (A * s) + a
        S[:,i+1] = s


    S = S.T
    S = np.array(S)

    x_coords = S[:,0]
    y_coords = S[:,1]
    z_coords = S[:,2]


    ax.plot(x_coords, y_coords, z_coords,
        '-k', label='Blind trajectory')



    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?

    # make B
    B = np.zeros([6,6])
    B = np.matrix(B)

    B[0,0] = bx
    B[1,1] = by
    B[2,2] = bz
    B[3,3] = bvx
    B[4,4] = bvy
    B[5,5] = bvz

    # make C
    C = np.matrix(np.zeros([3,6]))
    C[0,0] = rx
    C[1,1] = ry
    C[2,2] = rz

    # Initial conditions for s0 and Sigma0
    s = np.matrix([0, 0, 2, 15, 3.5, 4.0]).T

    Sigma = np.matrix(np.zeros([6,6]))
    for i in range(6):
        Sigma[i,i] = 1
    Sigma = 0.01 * Sigma

    # will store in S again
    S = np.zeros([6,K])
    S = np.matrix(S)
    S[:,0] = s


    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    # Defining functions

    # Eq(2)
    def predictS(sk):
        return (A * sk) + a

    # Eq(3)
    def predictSig(sigk):
        return ((A * sigk * A.T) + (B * B.T)).I

    # Eq(4)
    def updateSig(approx_sig):
        return (approx_sig + (C.T * C)).I

    # Eq(5)
    def updateS(sigk, approx_sig, approx_s, mk):
        return sigk * ((approx_sig * approx_s) + (C.T * mk))


    # putting steps all together
    i = 1
    while i < K:
        # predict
        approx_s = predictS(s)
        approx_Sigma = predictSig(Sigma)

        # update
        Sigma = updateSig(approx_Sigma)
        approx_Xk = np.matrix(approx_Xk)
        curr_mk = robot_mk[i,:].T

        s = updateS(Sigma, approx_Sigma, approx_s, curr_mk)

        # record
        S[:,i] = s

        # step
        i = i + 1

    # graph it
    S = S.T
    S = np.array(S)
    x_coords = S[:,0]
    y_coords = S[:,1]
    z_coords = S[:,2]


    ax.plot(x_coords, y_coords, z_coords,
        '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
