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
    s_true = np.loadtxt("P4_trajectory.txt", delimiter = ",")
   
    #####################

    x_coords = s_true[:,0]
    y_coords = s_true[:,1]
    z_coords = s_true[:,2]
   
    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')
    

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    m = np.loadtxt("P4_measurements.txt", delimiter = ",")
    ###### print type(m)    
    m_trans = np.transpose(m)

    Cvtr = np.array([ [1/rx,0,0], [0,1/ry,0], [0,0,1/rz] ])

    ppts = np.dot(Cvtr, m_trans)
    ppts_trans = np.transpose(ppts)
    #####################
     
    x_coords = ppts_trans[:,0]
    y_coords = ppts_trans[:,1]
    z_coords = ppts_trans[:,2]
    ax.plot(x_coords, y_coords, z_coords,
            '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    A = np.array([ [1,0,0,dt,0,0],
                   [0,1,0,0,dt,0],
                   [0,0,1,0,0,dt],
                   [0,0,0,1-c*dt,0,0],
                   [0,0,0,0,1-c*dt,0],
                   [0,0,0,0,0,1-c*dt] ])
    a = np.array([ [0,0,0,0,0,g*dt] ]).T
    s0 = np.array([ [0,0,2,15,3.5,4.0] ]).T
    
    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
    sk = s0
    s  = s0

    i = 1
    while i < K:

       sk = np.dot(A, sk) + a
       s  = np.c_[s, sk]
  
       i += 1

    ###### print s.shape  
    x_coords = s[0,:] 
    y_coords = s[1,:]
    z_coords = s[2,:]
    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.array([ [bx,0,0,0,0,0],
                   [0,by,0,0,0,0],
                   [0,0,bz,0,0,0],
                   [0,0,0,bvx,0,0],
                   [0,0,0,0,bvy,0],
                   [0,0,0,0,0,bvz] ])
    
    C = np.array([ [rx,0,0,0,0,0],
                   [0,ry,0,0,0,0],
                   [0,0,rz,0,0,0] ])
    

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)


    ###### s0 is the same as in Part3
    s0 = np.array([ [0,0,2,15,3.5,4.0] ]).T
    Sig0 = np.identity(6)*0.01
    
    def predictS(s_k):
        s_tilde = np.dot(A,s_k)+a
        return s_tilde
 
    def predictSig(Sig_k):
        Sig_tilde = np.linalg.inv( np.dot( np.dot(A,Sig_k),(A.T) ) +
                                   np.dot(B,(B.T)) )
        return Sig_tilde

    def updateSig(Sig_tilde):
        Sig_k1 = np.linalg.inv( Sig_tilde + np.dot((C.T),C) )
        return Sig_k1

    def updateS(Sig_k1,Sig_tilde,s_tilde,m_k1):
        s_k1 = np.dot( Sig_k1 , np.dot(Sig_tilde,s_tilde) +
                                (np.matrix( np.dot((C.T),m_k1) )).T 
                     )
        return s_k1




    myS = s0
    sk  = s0
    Sigk= Sig0
	
    for i in range(1,K):

	s_inter   = predictS(sk)
	Sig_inter = predictSig(Sigk)
	Sigk      = updateSig(Sig_inter)
	sk        = updateS(Sigk,Sig_inter,s_inter,m_trans[:,i])
	myS = np.c_[myS, sk]
    
    myS = np.squeeze(np.asarray(myS))


    #print myS
    x_coords = myS[0,:]
    y_coords = myS[1,:]
    z_coords = myS[2,:]
    ax.plot(x_coords, y_coords, z_coords, '-r', label='Filtered trajectory')





    # Show the plot
    ax.legend()
    plt.show()
