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
    # Read trajectory data into s_true
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',')

    # Plot the positions x^k
    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2],
             '--b', label='True trajectory')


    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    # Read measurements data into s_mes
    s_mes = np.loadtxt('P4_measurements.txt', delimiter=',')

    # Compute approximate position points
    strech = np.matrix([[1/rx,0,0], [0,1/ry,0], [0,0,1/rz]])
    x_aprox = strech*s_mes.transpose()
    x_aprox = np.asarray(x_aprox)

    # Plot the approximate position points
    ax.plot(x_aprox[0,:], x_aprox[1,:], x_aprox[2,:],
             '.g', label='Observed trajectory')


    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    # Initialize parameters A,s,a
    A = np.asmatrix(np.zeros([6,6]))
    A[0,0] = 1; A[1,1] = 1; A[2,2] = 1;
    A[0,3] = dt; A[1,4] = dt; A[2,5] = dt;
    A[3,3] = 1 - c*dt; A[4,4] = 1 - c*dt; A[5,5] = 1 - c*dt;
    a = np.matrix([0,0,0,0,0,g*dt]).transpose()
    s = np.asmatrix(np.zeros([6,K]))

    # Initial conditions for s0
    s_0 = np.matrix('0,0,2,15,3.5,4.0').transpose()

    # Compute the rest of sk using Eq (1)
    s[:,0] = s_0 # fill out first column
    i = 1
    while i < K:
        s[:,i] = A*s[:,i-1] + a
        i += 1
    
    # Plot approximate position points
    s = np.asarray(s)
    ax.plot(s[0,:], s[1,:], s[2,:],
             '-k', label='Blind trajectory')


    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    # Set matrix B and C
    B = np.asmatrix(np.zeros([6,6]))
    B[0,0] = bx; B[1,1] = by; B[2,2] = bz;
    B[3,3] = bvx; B[4,4] = bvy; B[5,5] = bvz;
    # cancatenate inverse of strech and 3*3 zero matrix
    C = np.hstack((np.linalg.inv(strech), np.asmatrix(np.zeros([3,3]))))

    # Initial conditions for s0 and Sigma0
    Sigma0 = 0.01*np.identity(6)

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    sKal = np.asmatrix(np.zeros([6,K]))
    sKal[:,0] = s_0 # fill out first column
    s_mes = np.asmatrix(s_mes)
    i = 0; Sigma = Sigma0; s = s_0;
    
    while i < K-1:
        # Predict
        s_apx = A*s + a
        Sigma_apx = np.linalg.inv(A*Sigma*A.transpose() + B*B.transpose())
        # Update
        Sigma = np.linalg.inv(Sigma_apx + C.transpose()*C)
        s = Sigma*(Sigma_apx*s_apx + C.transpose()*s_mes[i+1,:].transpose())
        sKal[:,i+1] = s
        i += 1

    # Plot the Kalman filter trajectory
    sKal = np.asarray(sKal)
    ax.plot(sKal[0,:], sKal[1,:], sKal[2,:],
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
