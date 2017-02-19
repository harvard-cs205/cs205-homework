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
    
    s_true = np.loadtxt('P4_trajectory.txt',delimiter=',')
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

    s_m = np.loadtxt('P4_measurements.txt',delimiter=',')
    m = s_m
    c_mat = np.array([[1/rx,0,0],[0,1/ry,0],[0,0,1/rz]])
    s_approx = np.dot(s_m,c_mat)
    x_coords = s_approx[:,0]
    y_coords = s_approx[:,1]
    z_coords = s_approx[:,2]  

    ax.plot(x_coords, y_coords, z_coords,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    A = np.matrix(
	[[1,0,0,dt,0,0],[0,1,0,0,dt,0],
	[0,0,1,0,0,dt],[0,0,0,1-c*dt,0,0],
	[0,0,0,0,1-c*dt,0],[0,0,0,0,0,1-c*dt]])
    a = np.matrix([[0,0,0,0,0,g*dt]]).transpose()
    s = np.zeros((6,K))

    # Initial conditions for s0
    s[:,0] = [0,0,2,15,3.5,4.0]
    s = np.matrix(s)
    
    # Compute the rest of sk using Eq (1)
    for x in xrange(1,K):
      s[:,x] = np.dot(A,s[:,x-1])+a
    s = np.squeeze(np.asarray(s))
    x_coords = s[0]
    y_coords = s[1]
    z_coords = s[2]  			
      
    ax.plot(x_coords, y_coords, z_coords,
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    A = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],
    [0,0,1,0,0,dt],[0,0,0,1-c*dt,0,0],
    [0,0,0,0,1-c*dt,0],[0,0,0,0,0,1-c*dt]])
    a = np.array([0, 0, 0, 0, 0, g * dt])
    B = np.diag([bx,by,bz,bvx,bvy,bvz])
    C = np.zeros((3,6))
    C[0,0] = rx
    C[1,1] = ry
    C[2,2] = rz

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    s = np.zeros((6,K))

    #initial s
    s[:,0] = np.array([0,0,2,15,3.5,4.0])
    
    #initialize Sigma
    sigma = 0.01 * np.identity(6)
    
    # Returns s_t
    def predictS(s_k):
      return np.add(np.dot(A,s_k), a)

    # Returns sigma_t
    def predictSig(sigma_k):
      return np.linalg.inv(np.add(np.dot(np.dot(A, sigma_k), A.T), np.dot(B, B.T)))
    
    # Returns sigma_k+1
    def updateSig(sigma_t):
      return np.linalg.inv(np.add(sigma_t, np.dot(C.T, C)))

    # Returns s_k+1
    def updateS(sigma,sigma_t,st,mk1):
      return np.dot(sigma, np.add(np.dot(sigma_t, s_t), np.dot(C.T, mk1)))
    
    for k in xrange(0, K - 1):
        s_t = predictS(s[:,k])
        sigma_t = predictSig(sigma)
        sigma = updateSig(sigma_t)
        s[:, k + 1] = updateS(sigma, sigma_t, s_t, m[k + 1])  

    x_coords = s[0]
    y_coords = s[1]
    z_coords = s[2]  
    ax.plot(x_coords, y_coords, z_coords,
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.savefig('P4.png', format='png')
    plt.show()

