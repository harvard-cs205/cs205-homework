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
    x_coords, y_coords, z_coords = np.loadtxt('P4_trajectory.txt', 
        delimiter=',', usecols=range(3), unpack=True)


    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    Cshort = np.asarray([[1/rx,0,0],[0,1/ry,0],[0,0,1/rz]])
    X = np.loadtxt('P4_measurements.txt', 
        delimiter=',', usecols=range(3), unpack=True)
    X = Cshort.dot(X)
    x_coords = X[0,:]
    y_coords = X[1,:]
    z_coords = X[2,:]
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.asmatrix([
        [1,0,0,dt,0,0],
        [0,1,0,0,dt,0],
        [0,0,1,0,0,dt],
        [0,0,0,1-c*dt,0,0],
        [0,0,0,0,1-c*dt,0],
        [0,0,0,0,0,1-c*dt]
        ])
    a = np.transpose(np.asmatrix([0,0,0,0,0,g*dt]))

    # Initial conditions for s0
    s = np.transpose(np.asmatrix([0, 0, 2, 15, 3.5, 4.0]))

    # Compute the rest of sk using Eq (1)
    T = np.zeros((6,K))
    for k in xrange(K) :
        T[:,k] = np.transpose(s)
        s = A * s + a

    x_coords = T[0,:]
    y_coords = T[1,:]
    z_coords = T[2,:]

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    B = np.asmatrix([
        [bx,0,0,0,0,0],
        [0,by,0,0,0,0],
        [0,0,bz,0,0,0],
        [0,0,0,bvx,0,0],
        [0,0,0,0,bvy,0],
        [0,0,0,0,0,bvz]
        ])
    C = np.asarray([
        [1/rx,0,0,0,0,0],
        [0,1/ry,0,0,0,0],
        [0,0,1/rz,0,0,0]
        ])

    def predictS(sk) :
        return A * sk + a

    def predictSig(sigmak) :
        inner = A * sigmak * np.transpose(A) + B.dot(np.transpose(B))
        return np.linalg.inv(inner)

    def updateSig(sigmap) :
        return np.linalg.inv(sigmap + np.transpose(C).dot(C))

    def updateS(sigmak, sigmap, sp, mk) :
        inner = sigmap.dot(sp) + np.transpose(C).dot(mk)
        return sigmak.dot(inner)

    X = np.asmatrix(np.loadtxt('P4_measurements.txt', 
        delimiter=',', usecols=range(3), unpack=True))

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    # Compute the rest of sk using Eq (1)
   
     # Initial conditions for s0
    s = np.transpose(np.asmatrix([0, 0, 2, 15, 3.5, 4.0]))
    sigma = 0.01*np.identity(6)
    print sigma
    T = np.zeros((6,K))
    for k in xrange(K) :
        T[:,k] = np.transpose(s)
        if(k + 1 == K) :
            break
        s = predictS(s)
        sigmap = predictSig(sigma)
        # sigma = updateSig(sigmap)
        # m = X[:,k+1]
        # s = updateS(sigma, sigmap, sp, m)


    x_coords = T[0,:]
    y_coords = T[1,:]
    z_coords = T[2,:]

    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
