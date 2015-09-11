import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def predictS(A, s_k, a):
    return np.dot(A, s_k) + a
def predictSig(A, Sig_k, B):
    return np.linalg.inv(A*Sig_k*(A.transpose()) + B*(B.transpose()))
def updateSig(sig, C):
    return np.linalg.inv(sig + (C.transpose())*C)
def updateS(sig_k1, sig, s, C, m):
    return sig_k1*(sig*s + (C.transpose())*m)

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

    s_true = np.loadtxt("P4_trajectory.txt", delimiter=',')

    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2],
            '--b', label='True trajectory')
    
    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    s_measured = np.loadtxt("P4_measurements.txt", delimiter=',')
    ax.plot((1/rx)*s_measured[:, 0], (1/ry)*s_measured[:,1], 
            (1/rz)*s_measured[:,2], '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    # Create each of the parts of the matrix
    top_left = np.diag(np.ones(3))
    top_right = np.diag(dt*np.ones(3))
    bottom_left= np.zeros((3,3))
    bottom_right = np.diag((1-c*dt)*np.ones(3))
    # Combine these into a single matrix
    A = np.hstack((np.vstack((top_left, bottom_left)), 
                   np.vstack((top_right, bottom_right))))
    a = np.vstack((np.matrix(np.zeros(5)).transpose(), np.array([g*dt])))
    s = np.matrix([0,0,2,15,3.5,4.0]).transpose()
    # Initial conditions for s0
    s_current = s

    # Compute the rest of sk using Eq (1)
    for i in range(K):
        s_next = np.dot(A,s_current) + a
        s = np.hstack((s, s_next))
        s_current = s_next
    s = np.array(s)
    ax.plot(s[0], s[1], s[2], '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.diag([bx, by, bz, bvx, bvy, bvz])
    C = np.matrix(np.hstack((np.diag([rx, ry, rz]), np.zeros((3,3)))))
    m = np.matrix(s_measured)
    
    # Initial conditions for s0 and Sigma0
    s = np.matrix([0,0,2,15,3.5,4.0]).transpose()
    s_current = s
    sigma_current = .01*np.matrix(np.identity(6))
    
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    for i in range(K):
        s_pred = predictS(A, s_current, a)
        sig_pred = predictSig(A, sigma_current, B)
        sig_next = updateSig(sig_pred, C)
        s_next = updateS(sig_next, sig_pred, s_pred, C, 
                np.matrix(m[i]).transpose())
        s = np.hstack((s, s_next))
        s_current = s_next
        sig_current = sig_next
    s = np.array(s)
    ax.plot(s[0], s[1], s[2], '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
