import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def predictS(A, sk, a):
    return (A * sk) + a

def predictSig(A, sigmak, B):
    return np.linalg.inv((A * sigmak * (A.transpose())) + (B * B.transpose()))

def updateSig(sigmatilda, C):
    return np.linalg.inv(sigmatilda + ((C.transpose()) * C))

def updateS(sigmakplus1, sigmatilda, stilda, C, mkplus1):
    return sigmakplus1 * ((sigmatilda * stilda) + ((C.transpose()) * mkplus1))

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

    Cinv = np.matrix([
        [1/rx, 0, 0], 
        [0, 1/ry, 0], 
        [0, 0, 1/rz]
    ])
    xtilda = Cinv * (m.transpose())
    xtilda = np.asarray(xtilda.transpose())

    ax.plot(xtilda[:,0], xtilda[:,1], xtilda[:,2],
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    
    A = np.matrix([
        [1, 0, 0, dt,     0,      0     ],
        [0, 1, 0, 0,      dt,     0     ],
        [0, 0, 1, 0,      0,      dt    ],
        [0, 0, 0, 1-c*dt, 0,      0     ],
        [0, 0, 0, 0,      1-c*dt, 0     ],
        [0, 0, 0, 0,      0,      1-c*dt]
    ])
    a = np.matrix([0, 0, 0, 0, 0, g*dt]).transpose()
    s = np.asmatrix(np.zeros([6, K]))
    
    # Initial conditions for s0
    s[:,0] = np.matrix([0, 0, 2, 15, 3.5, 4.0]).transpose()

    # Compute the rest of sk using Eq (1)
    i = 1
    while i < K:
        s[:,i] = A * s[:,i-1] + a
        i = i + 1
    s = np.asarray(s)

    ax.plot(s[0,:], s[1,:], s[2,:],
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.matrix([
        [bx, 0, 0, 0,  0,   0  ],
        [0, by, 0, 0,  0,   0  ],
        [0, 0, bz, 0,  0,   0  ],
        [0, 0, 0, bvx, 0,   0  ],
        [0, 0, 0, 0,   bvy, 0  ],
        [0, 0, 0, 0,   0,   bvz]
    ])
    C = np.matrix([
        [rx, 0, 0, 0, 0, 0], 
        [0, ry, 0, 0, 0, 0], 
        [0, 0, rz, 0, 0, 0]
    ])

    # Initial conditions for s0 and Sigma0
    s = np.asmatrix(np.zeros([6, K]))
    s[:,0] = np.matrix([0, 0, 2, 15, 3.5, 4.0]).transpose()
    
    sigma = np.asmatrix(np.eye(6)) * .01
    
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    i = 1
    while i < K:
        stilda = predictS(A, s[:,i-1], a)
        sigmatilda = predictSig(A, sigma, B)
        sigma = updateSig(sigmatilda, C)
        s[:,i] = updateS(sigma, sigmatilda, stilda, C, np.asmatrix(m[i,:]).transpose())
        i = i + 1
    s = np.asarray(s)

    ax.plot(s[0,:], s[1,:], s[2,:],
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()