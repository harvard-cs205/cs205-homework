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

    x_coords, y_coords, z_coords, v_x, v_y, v_z \
        = np.loadtxt('P4_trajectory.txt', delimiter = ',', unpack=True)

    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    x_coords, y_coords, z_coords = np.loadtxt('P4_measurements.txt', delimiter = ',', unpack=True)
    x_coords /= rx
    y_coords /= ry
    z_coords /= rz
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix([
        [1, 0, 0, dt, 0, 0], 
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt], 
        [0, 0, 0, 1 - c*dt, 0, 0], 
        [0, 0, 0, 0, 1 - c*dt, 0], 
        [0, 0, 0, 0, 0, 1 - c*dt]
        ])
    a = np.matrix([0., 0., 0., 0., 0., g*dt]).T
    s = np.matrix([0., 0., 2, 15, 3.5, 4.0]).T

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    sk = [s]
    xk = [s.item((0,0))]
    yk = [s.item((1,0))]
    zk = [s.item((2,0))]

    prev = s 
    for k in range(K-1): 
        s_next = A * prev + a
        sk.append(s_next)
        xk.append(s_next.item((0,0)))
        yk.append(s_next.item((1,0)))
        zk.append(s_next.item((2,0)))
        prev = s_next

    ax.plot(xk, yk, zk,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    def predictS(s_k): 
        return A * s_k + a 

    def predictSig(S): 
        return np.linalg.inv(A * S * A.T + B * B.T)

    def updateSig(S): 
        return np.linalg.inv(S + C.T * C)

    def updateS(S_kp1, S, s, m_kp1): 
        return S_kp1 * (S*s + C.T*m_kp1)

    
    B = np.diag([bx, by, bz, bvx, bvy, bvz])
    C = np.matrix([
        [rx, 0, 0, 0, 0, 0], 
        [0, ry, 0, 0, 0, 0], 
        [0, 0, rz, 0, 0, 0]
        ])


    m = [np.matrix([x_coords[i], y_coords[i], z_coords[i]]).T for i in range(K)]

    print "Length: ", len(m)

    Sigma = np.diag([0.01 for _ in range(6)])
    s = np.matrix([0., 0., 2, 15, 3.5, 4.0]).T

    xs = [s.item((0,0))] 
    ys = [s.item((1,0))] 
    zs = [s.item((2,0))]

    ss = [s] 
    for k in range(1, K):
        s_pred = predictS(s)
        assert s_pred.shape == (6,1)
        #print "S_pred", s_pred

        Sigma_pred = predictSig(Sigma)
        assert Sigma_pred.shape == (6,6)

        #print "Sigma_pred", Sigma_pred

        Sigma = updateSig(Sigma_pred)
        assert Sigma.shape == (6,6)

        #print "Sigma", Sigma

        s = updateS(Sigma, Sigma_pred, s_pred, m[k])
        ss.append(s)
        xs.append(s.item((0,0)))
        ys.append(s.item((1,0)))
        zs.append(s.item((2,0)))

        break
    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    ax.plot(xs, ys, zs,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
