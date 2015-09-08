import numpy as np
import numpy.matlib as matlib
import numpy.linalg as linalg
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

    #commented since writeup states a required s_true variable
    #x_coords, y_coords, z_coords, x_velo, y_velo, z_velo = np.loadtxt('P4_trajectory.txt', delimiter=',', unpack=True)

    s_true = np.loadtxt('P4_trajectory.txt', delimiter=',', dtype={'names': ('x_coords', 'y_coords', 'z_coords', 'x_velo', 'y_velo', 'z_velo'), 'formats': ('f8', 'f8', 'f8', 'f8', 'f8', 'f8')})
    x_coords = s_true['x_coords']
    y_coords = s_true['y_coords']
    z_coords = s_true['z_coords']    
    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    x_coords, y_coords, z_coords = np.loadtxt('P4_measurements.txt', delimiter=',', unpack=True)

    #create a matrix of the measurements for part 5
    measurements = np.matrix([x_coords, y_coords, z_coords])
    
    x_coords = x_coords/rx
    y_coords = y_coords/ry
    z_coords = z_coords/rz    
    

    ax.plot(x_coords, y_coords, z_coords,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix([[1., 0., 0., dt, 0., 0.], 
                   [0., 1., 0., 0., dt, 0.],
                   [0., 0., 1., 0., 0., dt],
                   [0., 0., 0., 1 - c*dt, 0., 0.],
                   [0., 0., 0., 0., 1 - c*dt, 0.],
                   [0., 0., 0., 0., 0., 1 - c*dt]]) 
    a = np.transpose(np.matrix([0.,0.,0.,0.,0.,g*dt]))
    s = matlib.zeros((6,K))

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    s[:, 0] = np.transpose(np.matrix([0.,0.,2.,15.,3.5,4.]))
    for x in range(1, K):
        s[:,x] = A*s[:,x-1] + a

    x_coords = np.asarray(s[0])[0]
    y_coords = np.asarray(s[1])[0]
    z_coords = np.asarray(s[2])[0]
    ax.plot(x_coords, y_coords, z_coords,
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.matrix([[bx, 0., 0., 0., 0., 0.], 
                   [0., by, 0., 0., 0., 0.],
                   [0., 0., bz, 0., 0., 0.],
                   [0., 0., 0., bvx, 0., 0.],
                   [0., 0., 0., 0., bvy, 0.],
                   [0., 0., 0., 0., 0., bvz]]) 
    C = np.matrix([[rx, 0., 0., 0., 0., 0.], 
                   [0., ry, 0., 0., 0., 0.],
                   [0., 0., rz, 0., 0., 0.]]) 
                   
    def predictS(s_k):
        return A*s_k + a
    
    def predictSig(cov_k):
        return linalg.inv(A*cov_k*np.transpose(A) + B*np.transpose(B))
        
    def updateSig(covsig_k):
        return linalg.inv(covsig_k + matlib.transpose(C)*C)
        
    def updateS(updateSig_k, predictSig_k, predictS_k, m):
        return updateSig_k*(predictSig_k * predictS_k + matlib.transpose(C)*m)

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    updateSig_k = np.identity(6)*.01
    s2 = np.matrix.copy(s)
    for x in range(1,K):
        predictS_k = predictS(s2[:,x-1])
        predictSig_k = predictSig(updateSig_k)
        updateSig_k = updateSig(predictSig_k)
        s2[:, x] = updateS(updateSig_k, predictSig_k, predictS_k, measurements[:, x])

    x_coords = np.asarray(s2[0])[0]
    y_coords = np.asarray(s2[1])[0]
    z_coords = np.asarray(s2[2])[0]
    ax.plot(x_coords, y_coords, z_coords,
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
