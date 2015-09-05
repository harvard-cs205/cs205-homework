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
    s_true=np.loadtxt('P4_trajectory.txt',delimiter=',')
    #Get transpose
    s_true=s_true.transpose()
    ax.plot(s_true[0], s_true[1], s_true[2],'--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    m_k=np.loadtxt('P4_measurements.txt',delimiter=',')
    #Set up Matrix C
    C=np.identity(3)
    C[0][0]=1/rx
    C[1][1]=1/ry
    C[2][2]=1/rz
    Col_length=len(m_k[:,0])
    x_k=np.zeros([Col_length,3])
    i=0
    while i<Col_length:
     x_k[i]=np.dot(m_k[i],C)
     i=i+1
    x_k=np.transpose(x_k)
    ax.plot(x_k[0], x_k[1], x_k[2], '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.zeros([6,6])
    A[0][0]=A[1][1]=A[2][2]=1
    A[0][3]=A[1][4]=A[2][5]=dt
    A[3][3]=A[4][4]=A[5][5]=1-c*dt
    
    a = [0,0,0,0,0,g*dt]
 
    s = np.array([0,0,2,15,3.5,4.0])
    # Initial conditions for s0
    S=np.zeros([6,K+1])
    S[:,0]=s
    # Compute the rest of sk using Eq (1)
    k=0
    while k<K:
      S[:,k+1]=np.dot(A,S[:,k])+a
      k=k+1
    
    ax.plot(S[0], S[1], S[2],'-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.zeros([6,6])
    B[0][0]=bx;B[1][1]=by;B[2][2]=bz
    B[3][3]=bvx;B[4][4]=bvy;B[5][5]=bvz
    
    C=np.zeros([3,6])
    C[0][0]=rx
    C[1][1]=ry
    C[2][2]=rz

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    # predictS
    def predictS(s,A,a):
      return np.dot(A,s)+a
    #predictSig
    def predictSig(A,Sig,B):
      return np.linalg.inv(A.dot(np.dot(Sig,A.transpose()))+np.dot(B,B.transpose()))
    #updateSig
    def updateSig(Sig_w,C):
      return np.linalg.inv(Sig_w+np.dot(C.transpose(),C))
    #updateS
    def updateS(Sig,s,C,m):
      return np.dot(np.linalg.inv(Sig+np.dot(C.transpose(),C)),(np.dot(Sig,s)+np.dot(C.transpose(),m)))
    #Init and Compute
    m=np.loadtxt('P4_measurements.txt',delimiter=',')
    S=np.zeros([6,K+1])
    s=S[:,0]=[0,0,2,15,3.5,4.0]
    Sig=0.01*np.identity(6)
    k=0
    while k<K:
     #s=predictS(s,A,a)
     s=updateS(predictSig(A,Sig,B),predictS(s,A,a),C,m[k,:])
     Sig=updateSig(predictSig(A,Sig,B),C)
     S[:,k+1]=s
     k=k+1
    ax.plot(S[0], S[1], S[2],
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
