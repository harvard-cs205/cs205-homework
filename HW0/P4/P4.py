import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv


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
    
    s_true = np.loadtxt("./P4_trajectory.txt", delimiter=',')
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
    s_measured = np.loadtxt("./P4_measurements.txt", delimiter=',')
    r = np.matrix(np.zeros([3,3]))
    np.fill_diagonal(r,[1/rx,1/ry,1/rz])
    
    x_approx = r*np.transpose(s_measured)
    x_coords = np.array(x_approx)[0]
    y_coords = np.array(x_approx)[1]
    z_coords = np.array(x_approx)[2] 
    
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?
    A = np.matrix(np.zeros([6,6]))
    np.fill_diagonal(A, [1,1,1,1 - c*dt,1 - c*dt,1 - c*dt])

    A[0,3] = dt
    A[1,4] = dt
    A[2,5] = dt

    a = np.matrix(np.zeros([6,1]))
    a[5,0] = g*dt

    s_k = np.matrix(np.zeros([6,len(s_measured)+1]))
 
    # Initial conditions for s0

    s_0 = np.matrix(np.array([0,0,2,15,3.5,4.0]))
    s_k[:,0]= s_0.T


    # Compute the rest of sk using Eq (1)
 
    for i in xrange(len(s_measured)):
    	s_k[:,i+1] = A*(s_k[:,i]) + a

    x_coords = np.array(s_k)[0].T
    y_coords = np.array(s_k)[1].T
    z_coords = np.array(s_k)[2].T
    
    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?
    B = np.matrix(np.zeros([6,6]))
    np.fill_diagonal(B,[bx,by,bz,bvx,bvy,bvz])

    C = np.matrix(np.zeros([3,6]))
    np.fill_diagonal(C,[rx,ry,rz])


    # Initial conditions for s0 and Sigma0
    sigma_0 = 0.01 * np.identity(6)

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    def predictS(s_k):
    	return A*(s_k) + a

    def predictSig(sigma):
    	return inv(A*sigma*A.T + B*B.T)

    def updateSig(sigma):
    	return inv(predictSig(sigma) + (C.T)*C)

    def updateS(sigma, s_k, m):
        return updateSig(sigma)*(predictSig(sigma)*predictS(s_k) + (C.T)*(m))

    s_k2 = np.matrix(np.zeros([6,len(s_measured)+1]))
    s_k2[:,0] = s_0.T

    s_measured = np.matrix(s_measured)

    for idx, i in enumerate(s_measured):	
    	if idx == 0:
        	sigma = sigma_0
    	else:
        	sigma = updateSig(sigma)
    	s_k2[:,idx+1] = updateS(sigma, s_k[:,idx], i.T)

    x_coords = np.array(s_k2)[0].T
    y_coords = np.array(s_k2)[1].T
    z_coords = np.array(s_k2)[2].T


    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
