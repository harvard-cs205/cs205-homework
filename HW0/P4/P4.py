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
    # load txt file
    s_true = np.loadtxt("P4_trajectory.txt", delimiter = ",")
    ax.plot(s_true[:,0],s_true[:,1] , s_true[:,2] ,'--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    s_measure = np.loadtxt("P4_measurements.txt", delimiter = ",")
    # construct the streching matrix
    s_streching = np.matrix([[1/rx,0,0], [0,1/ry,0],[0,0,1/rz]])
    # caliberate measurement by s_streching * s_measure
    s_caliberate = []
    for x in range(len(s_measure[:,0])):
	s_caliberate.append(np.dot(s_streching,s_measure[x].T))
    result = np.asarray(s_caliberate)
    ax.plot(result[:,0,0],result[:,0,1], result[:,0,2],'.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    #initialize constant matrix A,a,s	
    A = np.matrix([
	[1.0,0,0,dt,0,0],
	[0,1.0,0,0,dt,0],
	[0,0,1.0,0,0,dt],
	[0,0,0,1.0 - c * dt,0,0],
	[0,0,0,0,1.0 - c * dt,0],
	[0,0,0,0,0,1.0 - c * dt]
])
    a = np.matrix([
	[0,0,0,0,0, g * dt]
]).T
    s = np.array([
	[0.0,0.0,2.0,15.0,3.5,4.0]
])
    #print A,a,s
    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    sk = []
    sk.append(s)
    s0 = s.T
    for i in range(K):
	si = np.dot(A,s0) + a 
	sk.append(np.asarray(si).T)
        s0 = si
    
    #plot the coords for Blind trajectory 
    sk = np.asarray(sk)
    #print sk
    ax.plot(sk[:,0,0], sk[:,0,1], sk[:,0,2],
      '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # B = ?
    # C = ?
    # initialize matrix B and C
    b = np.zeros([6,6])
    B = np.asmatrix(b)
    B[0,0] = bx
    B[1,1] = by
    B[2,2] = bz
    B[3,3] = bvx
    B[4,4] = bvy
    B[5,5] = bvz
#    print "B",B

    c = np.zeros([3,6])
    C = np.asmatrix(c)
    C[0,0] = rx
    C[1,1] = ry
    C[2,2] = rz
#    print "C",C
    
    # Initial conditions for s0 and Sigma0
    s0 = s
    i = np.identity(6)
    I = np.asmatrix(i)
    Sig0 = 0.01 * I
#    print s0, Sig0
    
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    
    # define predictS implementing Eqs(2)
    def predictS(sk):
	return A * sk.T + a
  #  print predictS(s0)

    def predictSig(Sigk):
	temp = A*Sigk*A.T + B*B.T
	return np.linalg.inv(temp)
 #   print predictSig(Sig0)

    def updateSig(Sigk):
	temp = predictSig(Sigk) + C.T*C
	return np.linalg.inv(temp)
#    print updateSig(Sig0)
    
    mk = np.asmatrix(s_measure)
    def updateS(sk,Sigk,k_1):
	temp = updateSig(Sigk) * (predictSig(Sigk)*predictS(sk) + C.T * mk[k_1,:].T)
	return temp
#    print updateS(s0,Sig0,1)
   
    result = np.zeros([K,6]) 
    result[0] = s0
    s = s0
    Sig = Sig0
    for i in range( K-1):
	s1 = updateS(s, Sig,i+1).T
	result[i+1] = np.asarray(s1)
        s = s1
        Sig = updateSig(Sig)
       
#    print result
    ax.plot(result[:,0], result[:,1], result[:,2],'-r', label='Filtered trajectory')
    # Show the plot
    ax.legend()
    plt.show()
