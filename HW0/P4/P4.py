import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

#s,a are np.arrays and A is np.matrix
def predictS(A,s,a):
    #pdb.set_trace()
    lhs = np.dot(A,s)
    return np.array(lhs[0]) + a

def predictSig(A,sig,B):
    #pdb.set_trace()
    asa = np.dot(A,sig)
    asa = np.dot(asa,A.T)
    newB = np.dot(B,B.T)
    return np.linalg.inv(asa + newB)

def updateSig(Sig,C):
    #pdb.set_trace()
    return np.linalg.inv(Sig+np.dot(C.T,C))
#What is C???
def updateS(Sigk,Sig,s,C,m):
    #pdb.set_trace()
    rhsl = np.dot(Sig,s)
    rhsr = np.dot(C.T,m)
    return np.dot(Sigk, (rhsl + rhsr).T)

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
    fd = open('P4_trajectory.txt')

    
    s_true = []
    for line in fd:
        s_true.append(line[:-1].split(','))
    s_true = np.array(s_true)
    #print s_true[:,0]
    #print s_true[:10]
    ax.plot([float(x) for x in s_true[:,0]], [float(y) for y in s_true[:,1]],
     [float(z) for z in s_true[:,2]],'--b', label='True trajectory')
    #plt.show()
    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    fd2 = open('P4_measurements.txt')
    s_true2 = []
    #print fd2.readline().split(',')
    for line in fd2:
        s_true2.append(line[:-1].split(','))
    #print s_true2
    s_true2 = np.array(s_true2)
    ax.plot([float(x) for x in s_true2[:,0]],[float(y) for y in s_true2[:,1]],
        [float(z) for z in s_true2[:,2]],'.g',label='Observed trajectory')
    #plt.show()
    # ax.plot(x_coords, y_coords, z_coords,
    #         '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    # a = ?
    # s = ?
    cdt = 1-c*dt
    A = np.matrix([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],
        [0,0,0,cdt,0,0],[0,0,0,0,cdt,0],[0,0,0,0,0,cdt]])
    
    s = np.zeros((6,K))


    a = np.array([0,0,0,0,0,g*dt])

    s0 = np.array([0.0,0.0,2.0,15.0,3.5,4.0])
    s[:,0] = s0
    # print s
    #pdb.set_trace()
    for k in range(1,121):
        lhs = np.dot(A,s[:,k-1])
        result = np.array(lhs.T)[:,0]+a
        s[:,k] = result
    
    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    ax.plot(s[0], s[1], s[2],
            'k', label='Blind trajectory')
    #plt.show()
    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    
    B = np.matrix([[bx,0,0,0,0,0],[0,by,0,0,0,0],[0,0,bz,0,0,0]
        ,[0,0,0,bvx,0,0],[0,0,0,0,bvy,0],[0,0,0,0,0,bvz]])
    C = np.matrix([[rx,0,0,0,0,0],[0,ry,0,0,0,0],[0,0,rz,0,0,0]])
    
    # Initial conditions for s0 and Sigma0
    
    #sigmaMatrix = np.zeros(K)
    #sigmaMatrix[0] = Sigma0

    
    
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    
    KalmanSMat = np.zeros((6,K))
    KalmanSMat[:,0] = s0
    Sigma0 = np.identity(6)*0.01
    currentSigma = Sigma0
    
    for k in range(1,121):
        #pdb.set_trace()
        pS = predictS(A,KalmanSMat[:,k-1],a)
        pS = pS[0]#formatting 
        #pSigma = predictSig(A,sigmaMatrix[k-1],B)
        pSigma = predictSig(A,currentSigma,B)
        #sigmaMatrix[k] = updateSig(pSigma,C)
        currentSigma = updateSig(pSigma,C)
        #KalmanSMat[k] = updateS(sigmaMatrix[k],pSigma,pS,C,m)
        #pdb.set_trace()
        KalmanSMat[:,k] = np.array(updateS(currentSigma,pSigma,
            pS,C,[float(x) for x in s_true2[k]]))[:,0]
    #print KalmanSMat
    #pdb.set_trace()
    ax.plot(KalmanSMat[0], KalmanSMat[1], KalmanSMat[2],
            'r', label='Filtered trajectory')

    # Show the plot
    #ax.legend()
    plt.show()
