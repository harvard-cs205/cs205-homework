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
    
    def predictS(x,y,z):
        return np.dot(x,y)+z #to be called s_predicted in loop or sk
                         
    def predictSig(x,y,z):
        return inv(np.dot(np.dot(x,y),x.transpose())+np.dot(z,z.transpose())) #to be called Sig_predicted in loop or Sigk
    
    def updateSig(x,y):
        return inv(x+np.dot(y.transpose(),y)) #to be called Sig_updated in loop or Sig K+1
    
    def updateS(x,y,z,i,j):
        return np.dot(x,np.dot(y,z)+np.dot(i.transpose(),j)) #to be called s_new in loop or  Sk+1
    
    s0 = s0
    Sigma0 = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])*0.01
    B = np.array([[bx,0,0,0,0,0],[0,by,0,0,0,0],[0,0,bz,0,0,0],[0,0,0,bvx,0,0],[0,0,0,0,bvy,0],[0,0,0,0,0,bvz]])
    C = np.array([[1/rx,0,0,0,0,0],[0,1/ry,0,0,0,0],[0,0,1/rz,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
     
    s_new = np.zeros([6,K])
    i = 1
    s_new[:,0]=s0
    Sigma=Sigma0
    m_append=np.zeros([3,121])
    m=np.concatenate((m,m_append))
    
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    while i <= K-1:
        s_predicted = predictS(A,s_new[:,i-1],a)        
        Sig_predicted = predictSig(A,Sigma,B)
        Sig_updated = updateSig(Sig_predicted,C)
        s_new[:,i]=updateS(Sig_updated,Sig_predicted,s_predicted,C,m[:,i])        
        i+=1
        
    ax.plot(s_new[0,:], s_new[1,:], s_new[2,:],'-r', label='Filtered trajectory')
    # Show the plot
    ax.legend()
    plt.show()