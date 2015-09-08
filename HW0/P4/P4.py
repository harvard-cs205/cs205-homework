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
    #loaded the actual trajectory data using np.loadtxt function
    #indicated delimiter as a comma

    s_true = np.loadtxt('P4_trajectory.txt',delimiter=',')
    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2],'--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    #loaded measurement data using np.loadtxt like above
    #need to transpose matrix m to align with the mathematical formula
    #needed to perform dot multiplication to account for broadcasting error
    #plotted x which is the plot that takes into account dimensional stretching
    
    m = np.loadtxt('P4_measurements.txt',delimiter=',')
    m=m.transpose()    
    r = np.array([[1/rx,0,0],[0,1/ry,0],[0,0,1/ry]])
    x=np.dot(r,m)
    ax.plot(x[0,:],x[1,:],x[2,:],'.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    A = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1-c*dt,0,0],[0,0,0,0,1-c*dt,0,],[0,0,0,0,0,1-c*dt]])
    a = np.array([0,0,0,0,0,g*dt])
    a = a.transpose()
    # Initial conditions for s0
    s = np.zeros([6,K])
    s0=np.array([0,0,2,15,3.5,4])
    s0=s0.transpose()
    i = 1
    s[:,0]=s0
    
    # Compute the rest of sk using Eq (1)
    
    while i <= K-1:
        s[:,i]=np.dot(A,s[:,i-1])+a
        i+=1
    
    #Do i still need to convert to an array? appears to already be an ndarray
    
    ax.plot(s[0,:], s[1,:], s[2,:],'-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    
    B = np.array([[bx,0,0,0,0,0],[0,by,0,0,0,0],[0,0,bz,0,0,0],[0,0,0,bvx,0,0],[0,0,0,0,bvy,0],[0,0,0,0,0,bvz]])
    
    def predictS(s,A,a):
        return np.dot(A,s)+a                
        
    def predictSig(A,Cov,B):
        Sig_guess = np.dot(A,Cov)
        Sig_guess = np.dot(Sig_guess,A.transpose())
        Sig_guess = Sig_guess + np.dot(B,B.transpose())
        Sig_guess = Sig_guess**-1
    
    def updateSig(Sig_guess,C):
        
    def updateS():
    
    B = np.array([bx,0,0,0,0,0],[,0,by,0,0,0,0],[0,0,bz,0,0,0],[0,0,0,bvx,0,0],[0,0,0,0,bvy,0],[0,0,0,0,0,bvz])
    # C = ?

    # Initial conditions for s0 and Sigma0
    s0 = s0
    Sigma0 = np.array([[1,0,0],[0,1,0],[0,0,1]])*0.01
    

    s_new = np.zeros([6,K])
    i = 1
    s_new[:,0]=s0
    
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    while i <= K-1:
        s_new[:,i]=predictS(s_new[:,i-1],A,a)
        Sig_guess = predictSig(A,Cov,B)
        
        i+=1
        
    

    # ax.plot(x_coords, y_coords, z_coords,
    #         '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
