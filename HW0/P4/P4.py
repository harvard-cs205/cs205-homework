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
    trajectory_filename =  "P4_trajectory.txt"
    s_true = np.loadtxt(trajectory_filename,delimiter=',')
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
    measurements_filename = "P4_measurements.txt"
    measurements = np.loadtxt(measurements_filename,delimiter=',')
    C = np.matrix(np.zeros((3,6)))
    C[0,0] = rx
    C[1,1] = ry
    C[2,2] = rz
    C_inv = C.getI()
    x_approx = (C_inv * measurements.T).T
    ax.plot(x_coords, y_coords, z_coords,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Initial conditions for s0
    A = np.matrix([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,(1 - c * dt),0,0],[0,0,0,0,(1 - c * dt),0],[0,0,0,0,0,(1 - c * dt)]],dtype=float)
    a = np.transpose(np.array([[0,0,0,0,0,g*dt]]))
    s = np.transpose(np.array([[0,0,2,15,3.5,4]]))
    predicted_s = np.matrix(np.zeros((6,K)))
    predicted_s[:,0] = s
    s_k = s
    # Compute the rest of sk using Eq (1)
    for k in range(K-1):
        s_next = A * s_k + a
        predicted_s[:,k+1] = s_next
        s_k = s_next
    predicted_s_array = np.asarray(predicted_s)
    x_coords = predicted_s_array[0]
    y_coords = predicted_s_array[1]
    z_coords = predicted_s_array[2]
    ax.plot(x_coords, y_coords, z_coords,
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.zeros((6,6))
    B[0,0] = bx
    B[1,1] = by
    B[2,2] = bz
    B[3,3] = bvx
    B[4,4] = bvy
    B[5,5] = bvz
    # C is defined in Part 2

    # Initial conditions for s0 and Sigma0
    s0 = s
    Sigma0 = 0.01 * np.identity(6)
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    def predictS(s,A,a):
        '''
        Takes column vector s, propogation matrix A, and column vector a as parameters and returns the intermediate s at the next time step as a column vector
        '''
        return A*s+a
    
    def predictSig(A,Sigma,B):
        '''
        Takes the propagation matrix A, covariance matrix Sigma, and Error matrix B as parameters and returns the intermediateSigma at the next time step
        '''
        return np.linalg.inv(A*Sigma*A.getT()+B*np.transpose(B))
    
    def updateSig(Sigma_i,C): #Sigma_i is intermediate Sigma
        '''
        Takes the covariance matrix Sigma and the stretch-correcting matrix C as parameters and returns the updated Sigma
        '''
        return (Sigma_i + C.getT() * C).getI()
    
    def updateS(Sigma, Sigma_i, C, s_i, m):
        '''
        Takes matrices updated Sigma, intermediate Sigma_i, and stretch correcting C as parameters.  Also takes column vectors intermediate s_i, and measurement m.  Returns an update s prediction.
        '''
        return Sigma * (Sigma_i * s_i + C.getT()*m)
    
    updated_predicted_s = np.matrix(np.zeros((6,K)))
    updated_predicted_s[:,0] = s
    Sigma = Sigma0
    for k in range(K-1):
        s_i = predictS(s,A,a)
        Sigma_i = predictSig(A,Sigma,B)
        Sigma = updateSig(Sigma_i, C)
        m = np.resize(measurements[k], (3,1))
        s = updateS(Sigma,Sigma_i,C,s_i,m)
        updated_predicted_s[:,k+1] = s
    updated_predicted_s_array = np.asarray(updated_predicted_s)
    x_coords = updated_predicted_s_array[0]
    y_coords = updated_predicted_s_array[1]
    z_coords = updated_predicted_s_array[2]
    ax.plot(x_coords, y_coords, z_coords,
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
