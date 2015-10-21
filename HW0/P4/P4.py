import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as lin

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
    s_true=np.loadtxt('P4_trajectory.txt',delimiter=',')
    x_coords=s_true[:,0]
    y_coords=s_true[:,1]
    z_coords=s_true[:,2]

    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    measurements=np.loadtxt('P4_measurements.txt',delimiter=',')
    C=np.zeros((3,6))
    C[0,0]=1/rx
    C[1,1]=1/ry
    C[2,2]=1/rz
    appro=np.dot(measurements,C)
    x_coords=appro[:,0]
    y_coords=appro[:,1]
    z_coords=appro[:,2]
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.matrix([[1,0,0,dt,0,0], [0,1,0,0,dt,0], [0,0,1,0,0,dt], 
        [0,0,0,1-c*dt,0,0], [0,0,0,0,1-c*dt,0], [0,0,0,0,0,1-c*dt]])

    a = np.matrix([0,0,0,0,0,g*dt]).transpose()
    s = np.zeros((6,K))
    s0 = np.matrix([0,0,2,15,3.5,4.0])
    s[:,0] = s0
    s = np.asmatrix(s)

    for i in xrange(0,K-1):
        s[:,i+1]=np.add(np.dot(A,s[:,i]),a)

    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    s=np.asarray(s)
    ax.plot(s[0,:], s[1,:], s[2,:],
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.zeros((6,6))
    C = np.zeros((3,6))
    B[0,0],B[1,1],B[2,2],B[3,3],B[4,4],B[5,5]=bx,by,bz,bvx,bvy,bvz
    C[0,0],C[1,1],C[2,2]=rx,ry,rz
    B=np.asmatrix(B)
    C=np.asmatrix(C)
    
    Sigma=np.identity(6)*0.01

    def predictS(A,s,a):
        return np.dot(A,s)+a

    def predictSig(A,Sigma,B):
        return lin.inv(np.add(np.dot(np.dot(A,Sigma),A.transpose()),np.dot(B,B.transpose())))

    def updateSig(SigmaBar,C):
        return lin.inv(np.add(SigmaBar,np.dot(C.transpose(),C))) 

    def updateS(Sigma,SigmaBar,sBar,C,m):
        return np.dot(Sigma,np.add(np.dot(SigmaBar,sBar),np.dot(C.transpose(),m))) 

    s = np.zeros((6,K))
    s[:,0] = s0
    s = np.asmatrix(s)

    m=np.asmatrix(measurements).transpose()

    for i in xrange(0,K-1):
        s_predicted = predictS(A,s[:,i],a) 
        Sig_predicted = predictSig(A,Sigma,B)
        Sig_updated = updateSig(Sig_predicted,C)
        s[:,i+1]=updateS(Sig_updated,Sig_predicted,s_predicted,C,m[:,i])     

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    s=np.asarray(s)
    ax.plot(s[0,:], s[1,:], s[2,:],
            '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
