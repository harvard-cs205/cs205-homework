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
    x_coords = [sk[0] for sk in s_true]
    y_coords = [sk[1] for sk in s_true]
    z_coords = [sk[2] for sk in s_true]
    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    ms = np.loadtxt('P4_measurements.txt', delimiter=',')
    C = np.matrix([[1.0/rx, 0, 0], [0, 1.0/ry, 0], [0, 0, 1.0/rz]])
    xs = ms*C
    xs = np.array(xs)
    x_coords = [x[0] for x in xs]
    y_coords = [x[1] for x in xs]
    z_coords = [x[2] for x in xs]

    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.array([[1,0,0,dt,0,0], [0,1,0,0,dt,0], [0,0,1,0,0,dt], [0,0,0,1-c*dt,0,0], [0,0,0,0,1-c*dt,0], [0,0,0,0,0,1-c*dt]])
    a = np.array([0,0,0,0,0,g*dt]).T
    s = np.array([0,0,2,15,3.5,4.0]).T
    print A.shape, a.shape, s.shape
    pathmat = [0 for i in range(len(s_true))]
    pathmat[0] = s
    # print np.array(pathmat)
    for i in range(1, len(pathmat)):
        s = np.dot(A, s) + a
        pathmat[i] = s
    pathmat = np.array(pathmat).T
    x_coords = pathmat[0]
    y_coords = pathmat[1]
    z_coords = pathmat[2]


    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.array([[bx,0,0,0,0,0], [0,by,0,0,0,0], [0,0,bz,0,0,0], [0,0,0,bvx,0,0], [0,0,0,0,bvy,0], [0,0,0,0,0,bvz]])
    C = np.array([[rx,0,0,0,0,0], [0,ry,0,0,0,0], [0,0,rz,0,0,0]])
    sk = np.array([0,0,2,15,3.5,4.0]).T
    Sigma = 0.01*np.identity(6)
    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    def eq2(sk):
        return np.dot(A, sk) + a

    def eq3(Sigmak):
        return np.linalg.inv(np.dot(np.dot(A, Sigmak), A.T) + np.dot(B, B.T))

    def eq4(SigmaTilde):
        return np.linalg.inv(SigmaTilde+np.dot(C.T, C))

    def eq5(Sigmak, SigmaTilde, sTilde, mk):
        return np.dot(Sigmak, np.dot(SigmaTilde, sTilde)+np.dot(C.T, mk))
    
    pathmat = [0 for i in range(len(ms))]
    pathmat[0] = sk
    for i in range(1, len(ms)):
        sTilde = eq2(sk)
        SigmaTilde = eq3(Sigma)
        Sigmak = eq4(SigmaTilde)
        sk = eq5(Sigmak, SigmaTilde, sTilde, ms[i])
        pathmat[i] = sk
    pathmat = np.array(pathmat).T
    x_coords = pathmat[0]
    y_coords = pathmat[1]
    z_coords = pathmat[2] 

    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
