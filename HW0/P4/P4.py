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

    s_meas = np.loadtxt('P4_measurements.txt',delimiter=',')

    s_meas_positions = s_meas[:,:3] # just the position information, Nx3 array

    s_meas_positionsT = s_meas_positions.transpose() # convert to 3xN array

    R = np.array([[1/rx,0,0],[0,1/ry,0],[0,0,1/rz]]) # instrument matrix 3x3

    s_measCT = np.dot(R,s_meas_positionsT) # calibrate measurement
    s_measC = s_measCT.transpose() # transpose back, because otherwise plotting somehow doesn't work..
    
    x_coords = s_measC[:,0]
    y_coords = s_measC[:,1]
    z_coords = s_measC[:,2]

    ax.plot(x_coords, y_coords, z_coords,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1-c*dt,0,0],[0,0,0,0,1-c*dt,0],[0,0,0,0,0,1-c*dt]])
    a = np.array([0,0,0,0,0,g*dt]) 
    s0 = np.array([0,0,2,15,3.5,4.0])
    s = np.zeros([K,6])

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    s[0] = s0
    sk = s0

    i = 1
    while i<K:
        sk = A.dot(sk.transpose()).transpose() + a
        s[i] = sk
        i = i+1    

    x_coords = s[:,0]
    y_coords = s[:,1]
    z_coords = s[:,2]


    ax.plot(x_coords, y_coords, z_coords,'-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.array([[bx,0,0,0,0,0],[0,by,0,0,0,0],[0,0,bz,0,0,0],[0,0,0,bvx,0,0],[0,0,0,0,bvy,0],[0,0,0,0,0,bvz]])
    C = np.array([[rx,0,0,0,0,0],[0,ry,0,0,0,0],[0,0,rz,0,0,0]])

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    mx = s_meas[:,0]
    my = s_meas[:,1]
    mz = s_meas[:,2]
    sig0 = 0.01*np.identity(6)
    zz = np.zeros(mx.size)
    mm = np.array([mx.transpose(),my.transpose(),mz.transpose()])
    CC = C.transpose()
    skal = np.zeros([K,6])
    skal[0]= s0
    
    def predictS(x):
        return A.dot(x.transpose()).transpose()+a

    from numpy.linalg import inv    
    
    def predictSig(x):
        return inv(A.dot(x.dot(A.transpose()))+B.dot(B.transpose()))
    
    def updateSig(x):        
        return inv(x+CC.dot(C))
    
    def updateS(xsig,xsigexp,xs,xm):
        return xsig.dot(xsigexp.dot(xs)+CC.dot(xm))
       
    i=0
    while i<(K-1):
        sexp = predictS(skal[i])
        sigexp = predictSig(sig0)
        sig0 = updateSig(sigexp)
        i = i+1
        skal[i] = updateS(sig0,sigexp,sexp,mm[:,i])
        
    x_coords = skal[:,0]
    y_coords = skal[:,1]
    z_coords = skal[:,2]
  

    ax.plot(x_coords, y_coords, z_coords,
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
    
 
