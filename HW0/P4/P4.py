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
    x_coords, y_coords,z_coords, v_x, v_y, v_z = np.loadtxt('P4_trajectory.txt', 
        delimiter=',',unpack=True)

    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    x_coords, y_coords, z_coords = np.loadtxt('P4_measurements.txt', delimiter=',',unpack=True)
    ax.plot(x_coords/rx, y_coords/ry, z_coords/rz,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # given
    A = np.matrix([
        [1,0,0,dt,0,0],
        [0,1,0,0,dt,0],
        [0,0,1,0,0,dt],
        [0,0,0,1-c*dt,0,0],
        [0,0,0,0,1-c*dt,0],
        [0,0,0,0,0,1-c*dt]])
    a = np.matrix([[0],[0],[0],[0],[0],[g*dt]])
    s = np.matrix([[0],[0],[2],[15],[3.5],[4.0]])

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    # new matrix vectors
    x_coords2 = [s.item((0,0))]
    y_coords2 = [s.item((1,0))]
    z_coords2 = [s.item((2,0))]

    # iterate and update for x,y,z vals
    prev = s
    for _ in range(K-1):
        cur = A*prev+a
        x_coords2.append(cur.item((0,0)))
        y_coords2.append(cur.item((1,0)))
        z_coords2.append(cur.item((2,0)))
        prev = cur


    ax.plot(x_coords2, y_coords2, z_coords2,
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # given
    B = np.matrix([
        [bx,0,0,0,0,0],
        [0,by,0,0,0,0],
        [0,0,bz,0,0,0],
        [0,0,0,bvx,0,0],
        [0,0,0,0,bvy,0],
        [0,0,0,0,0,bvz]])
    C = np.matrix([
        [rx,0,0,0,0,0],
        [0,ry,0,0,0,0],
        [0,0,rz,0,0,0]])

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    # given
    def predictS(sk):
        return A*sk+a
    def predictSig(S):
        return np.linalg.inv(A * S * A.T + B * B.T)
    def updateSig(S):
        return np.linalg.inv(S + C.T*C)
    def updateS(Sk,St,st,mk):
        return Sk * (St * st + C.T * mk)

    m = [np.matrix([x_coords[k],y_coords[k],z_coords[k]]).T for k in range(K)]
    sig = np.matrix([
        [.1,0,0,0,0,0],
        [0,.1,0,0,0,0],
        [0,0,.1,0,0,0],
        [0,0,0,.1,0,0],
        [0,0,0,0,.1,0],
        [0,0,0,0,0,.1]])
    s = np.matrix([[0],[0],[2],[15],[3.5],[4.0]])

    x = [s.item((0,0))] 
    y = [s.item((1,0))]
    z = [s.item((2,0))]

    # filtering via equation
    for k in range(1,K):
        s_t = predictS(s)
        sig_t = predictSig(sig)
        sig = updateSig(sig_t)
        s = updateS(sig,sig_t,s_t,m[k])
        # add to list
        x.append(s.item((0,0)))
        y.append(s.item((1,0)))
        z.append(s.item((2,0)))

    ax.plot(x,y,z,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
