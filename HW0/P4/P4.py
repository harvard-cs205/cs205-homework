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
    
    

    s_true = np.loadtxt('P4_trajectory.txt',delimiter=',') #loaded the actual trajectory data using np.loadtxt function. indicated delimiter as a comma
    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2],'--b', label='True trajectory') #plot data

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    m = np.loadtxt('P4_measurements.txt',delimiter=',') #loaded measurement data using np.loadtxt like above
    m=m.transpose() #need to transpose matrix m to align with the mathematical formula / matrix multiiplication 
    r = np.array([[1/rx,0,0],[0,1/ry,0],[0,0,1/rz]]) #initialize r
    x=np.dot(r,m) #needed to perform dot multiplication to account for broadcasting error
    ax.plot(x[0,:],x[1,:],x[2,:],'.g', label='Observed trajectory')  #plotted x which is the plot that takes into account dimensional stretching

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    A = np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1-c*dt,0,0],[0,0,0,0,1-c*dt,0,],[0,0,0,0,0,1-c*dt]]) #initialize A array
    a = np.array([0,0,0,0,0,g*dt]) #initialize a array
    a = a.transpose() #transpose a
    s = np.zeros([6,K]) #initialize s vector
    s0=np.array([0.0,0.0,2.0,15.0,3.5,4.0]) #set s0
    s0=s0.transpose() #tranpose s0 to align into structure of how sk will be built
    i = 1
    s[:,0]=s0 #assign first column 0 of sk to be equal to s0
    
    # Compute the rest of sk using Eq (1)
    
    while i <= K-1:
        s[:,i]=np.add(np.dot(A,s[:,i-1]),a) #use np.add and np.dot like above to account for having arrays here
        i+=1
    
    ax.plot(s[0,:], s[1,:], s[2,:],'-k', label='Blind trajectory') #plot blind trajectory

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    
    def predictS(A,s,a):
        return np.add(np.dot(A,s),a) #to be called s_predicted in loop or sk
                         
    def predictSig(A,Sigma,B): #to be called Sig_predicted in loop or Sigk
        return inv(np.add(np.dot(np.dot(A,Sigma),A.transpose()),np.dot(B,B.transpose())))
        
    def updateSig(x,y):
        return inv(np.add(x,np.dot(y.transpose(),y))) #to be called Sig_updated in loop or Sig K+1
    
    def updateS(x,y,z,i,j):
        return np.dot(x,np.add(np.dot(y,z),np.dot(i.transpose(),j))) #to be called s_new in loop or  Sk+1
    
    s0 = s0
    Sigma0 = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])*0.01
    B = np.array([[bx,0,0,0,0,0],[0,by,0,0,0,0],[0,0,bz,0,0,0],[0,0,0,bvx,0,0],[0,0,0,0,bvy,0],[0,0,0,0,0,bvz]])
    C = np.array([[rx,0,0,0,0,0],[0,ry,0,0,0,0],[0,0,rz,0,0,0]])
     
    s_new = np.zeros([6,K])
    i = 1
    s_new[:,0]=s0
    Sigma=Sigma0
    #m_append=np.zeros([3,121])
    #m=np.concatenate((m,m_append))
    
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