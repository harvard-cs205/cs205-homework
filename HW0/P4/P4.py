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
    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',', dtype={'names':('x', 'y', 'z'), 'formats': (np.float, np.float, np.float)}, usecols = (0,1,2))
    #####################
    # ax.plot(x_coords, y_coords, z_coords,
    #         '--b', label='True trajectory
    x_coords = s_true['x']
    y_coords = s_true['y']
    z_coords = s_true['z']
  
    for s in s_true:
        ax.plot(x_coords, y_coords, z_coords, '--b', label='True trajectory')
        
    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
        
    m = np.loadtxt('P4_measurements.txt', delimiter=',', usecols = (0,1,2))
    #####################
    # ax.plot(x_coords, y_coords, z_coords,
    #         '.g', label='Observed trajectory')
    
    
    x_coords = 1. / rx * m[0]
    y_coords = 1. / ry * m[1]
    z_coords = 1. / rz * m[2]
    
    ax.plot(x_coords, y_coords, z_coords, '.g', label='Observed trajectory')
    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    # A = ?    
    # a = ?
    # s = ?
    A = np.matrix([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1-c*dt, 0, 0],
                  [0, 0, 0, 0, 1-c*dt, 0],
                  [0, 0, 0, 0, 0, 1-c*dt]])
                  
    a = np.matrix([0, 0, 0, 0, 0, g*dt])    

    sk = np.zeros((6,K))
   

    # Initial conditions for s0
    s0 = np.array([0, 0, 2, 15, 3.5, 4.0])
    sk[:,0]=s0
    s = np.dot(A,s0) + a
    # Compute the rest of sk using Eq (1)
    for x in xrange(1, K):
        sk[:,x] = s
        s = np.dot(A,sk[:,x]) + a
    
    # ax.plot(x_coords, y_coords, z_coords,
    #         '-k', label='Blind trajectory')
    
    x_coords = sk[0]
    y_coords = sk[1]
    z_coords = sk[2]
    
    ax.plot(x_coords, y_coords, z_coords, '-k', label='Blind trajectory')
    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?

    B = np.matrix([[bx, 0, 0, 0, 0, 0],
                  [0, by, 0, 0, 0, 0],
                    [0, 0, bz, 0, 0, 0],
                    [0, 0, 0, bvx, 0, 0],
                    [0, 0, 0, 0, bvy, 0],
                    [0, 0, 0, 0, 0, bvz]])
                    
    C = np.matrix([[rx, 0, 0, 0, 0, 0], [0, ry, 0, 0, 0, 0], [0, 0, rz, 0, 0, 0]])
    # Initial conditions for s0 and Sigma0
    s0 = np.array([[0], [0], [2], [15], [3.5], [4.0]])
    sigma0 = 0.01 * np.identity(6) 
    mt = m.transpose()

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    def predictS(s):
        return np.dot(A, s) + a

    def predictSig(sigma):
        pt1 = np.dot(np.dot(A,sigma),A.transpose())
        pt2 = np.dot(B,B.transpose())
        return np.linalg.inv(pt1+pt2)
        
    def updateSig(sigma):
        return  np.linalg.inv(sigma+np.dot(C.transpose(),C))
        
    def updateS(sigma,sigma0,s,mt):
        pt1 = sigma*(np.dot(sigma0,s.transpose()))
        pt2 = np.dot(mt,C)
        return pt1+pt2
        
    a = np.matrix([[0], [0], [0], [0], [0], [g*dt]])
    sk = np.zeros([K, 6])    
    sk[0] = s0.transpose()
    
    #sigma = predictSig(sigma0)  

    # Compute the rest of sk using Eq (1)
    for x in xrange(1, K):     
        sk[:x] = predictS(s0).transpose()
        sigma = predictSig(sigma0)
        sigma0 = updateSig(sigma)
        sk[:x] = updateS(sigma, sigma0, sk, m)
        s = sk[:x]
        sigma = sigma0
        
      
    x_coords = sk[0]
    y_coords = sk[1]
    z_coords = sk[2]
    
    # ax.plot(x_coords, y_coords, z_coords,
    #         '-r', label='Filtered trajectory')

    a.plot(x_coords, y_coords, z_coords, '-r', label='Filtered trajectory')
    # Show the plot
    ax.legend()
    plt.show()
