import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

A = np.empty([6,6])
A[0,0] = A[1,1] = A[2,2] = 1
A[0,3] = A[1,4] = A[2,5] = dt
A[3,3] = A[4,4] = A[5,5] = 1 - c*dt
a = np.array([0,0,0,0,0,g*dt])

C = np.empty([3,6])
C[0,0] = rx
C[1,1] = ry
C[2,2] = rz

def main():

    # Create 3D axes for plotting
    ax = Axes3D(plt.figure())

    #####################
    # Part 1:
    #
    # Load true trajectory and plot it
    # Normally, this data wouldn't be available in the real world
    #####################
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',')
    ax.plot([x[0] for x in s_true], [x[1] for x in s_true], [x[2] for x in s_true], '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    s_measured = np.loadtxt('P4_measurements.txt', delimiter=',')
    stch_mat = [[1/rx,0,0],[0,1/ry,0],[0,0,1/rz]]
    s_observed = [np.dot(stch_mat,x) for x in s_measured]
    ax.plot([x[0] for x in s_observed], [x[1] for x in s_observed], [x[2] for x in s_observed],'.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    s_k = [0,0,2,15,3.5,4.0]
    s = [s_k]
    for i in range(K-1):
        s_k1 = np.dot(A, s_k) + a
        s = np.vstack((s, s_k1))
        s_k = s_k1    

    ax.plot([x[0] for x in s], [x[1] for x in s], [x[2] for x in s], '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    # A, B, C are defined outside this block


    s_k = [0,0,2,15,3.5,4.0]
    Sig_k = np.identity(6)*0.01
    s = [s_k]

    for i in range(K-1):
        s_tilda = predictS(s_k)
        Sig_tilda = predictSig(Sig_k)
        Sig_k1 = updateSig(Sig_tilda)
        s_k1 = updateS(s_tilda, Sig_k1, Sig_tilda, s_observed[i+1][:3])
        # add to storage matrix
        s = np.vstack((s, s_k1))
        # update s_k and Sig_k
        s_k = s_k1
        Sig_k = Sig_k1


    ax.plot([x[0] for x in s], [x[1] for x in s], [x[2] for x in s], '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()


def predictS(s_k):
    s_tilda = np.dot(A, s_k) + a
    return s_tilda

def predictSig(Sig_k):
    B = np.empty([6,6])
    B[0,0] = bx
    B[1,1] = by
    B[2,2] = bz
    B[3,3] = bvx
    B[4,4] = bvy
    B[5,5] = bvz

    Sig_tilda = np.linalg.inv(np.dot(A,np.dot(Sig_k,np.transpose(A)))+np.dot(B,np.transpose(B)))
    return Sig_tilda

def updateSig(Sig_tilda):
    Sig_k1 = np.linalg.inv(Sig_tilda+np.dot(np.transpose(C),C))
    return Sig_k1

def updateS(s_tilda, Sig_k1, Sig_tilda, m_k1):
    S_k1 = np.dot(Sig_k1,(np.dot(Sig_tilda, s_tilda) + np.dot(np.transpose(C),m_k1)))
    return S_k1

if __name__ == '__main__':
    main()
