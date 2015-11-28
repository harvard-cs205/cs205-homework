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

    s_true = np.loadtxt('P4_trajectory.txt', delimiter = ',', unpack = True)
    ax.plot(s_true[0], s_true[1], s_true[2], '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    measurement = np.loadtxt('P4_measurements.txt', delimiter = ',', unpack = True)
    ax.plot(measurement[0] * (1/rx), measurement[1] * (1/ry), measurement[2] * (1/rz), '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Initial conditions for s0
    s0 = np.matrix([0.,0.,2.,15.,3.5,4]).T

    # Compute the rest of sk using Eq (1)
    a = np.matrix([0,0,0,0,0,g*dt]).T
    A = np.matrix(np.zeros([6,6]))
    A[0,0] = A[1,1] = A[2,2] = 1
    A[0,3] = A[1,4] = A[2,5] = dt
    A[3,3] = A[4,4] = A[5,5] = 1 - c*dt
    
    pred = np.matrix(np.zeros([6, K]))

    pred[:, 0] = s0
    for i in np.arange(K):
	if i == 0: pred[:, 0] = s0
	else: pred[:, i] = (A * pred[:, i-1]) + a
 
    pred = np.array(pred)
  
    ax.plot(pred[0], pred[1], pred[2], '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.matrix(np.identity(6))
    B *= np.matrix([bx, by, bz, bvx, bvy, bvz]).T

    C = np.matrix(np.zeros([3,6]))
    C[0,0], C[1,1], C[2,2] = rx, ry, rz

    # Initial conditions for s0 and Sigma0
    Sigma0 = 0.01*np.identity(6)
    
    Sigmak, sk, m = Sigma0, s0, np.matrix(measurement)

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    predictS = lambda A, sk, a: A*sk + a
    predictSig = lambda A, sigmak, B: (A*sigmak*A.T + B*B.T).I
    updateSig = lambda sigma, C: (sigma + C.T * C).I
    updateS = lambda sigmakp1, sigma, s, C, m: sigmakp1 * (sigma * s + C.T * m)

    kalman = np.matrix(np.zeros([6,K]))
    for i in np.arange(K):
	if i == 0: 
	    kalman[:, i] = sk
	else:
	    s = predictS(A, kalman[:, i-1], a)
	    sigma = predictSig(A, Sigmak, B)
	    sigmak = updateSig(sigma, C)
	    kalman[:, i] = updateS(sigmak, sigma, s, C, m[:, i])
	
    kalman = np.array(kalman)
    
    ax.plot(kalman[0, :], kalman[1, :], kalman[2, :], '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
