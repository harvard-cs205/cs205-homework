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
    s_true=np.loadtxt('P4_trajectory.txt',delimiter=',')
    
    # xt_coords,yt_coords,zt_coords=np.loadtxt('P4_trajectory.txt',delimiter=',',usecols=(0, 1, 2), unpack=True)
    x1_coords=s_true[:,0]
    y1_coords=s_true[:,1]
    z1_coords=s_true[:,2]

    #####################

    ax.plot(x1_coords, y1_coords, z1_coords, '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    s_m=np.loadtxt('P4_measurements.txt',delimiter=',')
    s_m=np.array(s_m)
     
    #xo_coords,yo_coords,zo_coords=np.loadtxt('P4_measurements.txt',delimiter=',', usecols=(0, 1, 2), unpack=True)
    s_obs=np.dot(s_m, np.diag([1/rx,1/ry,1/rz]))
    x2_coords=s_obs[:,0]
    y2_coords=s_obs[:,1]
    z2_coords=s_obs[:,2]

    ax.plot(x2_coords, y2_coords, z2_coords, '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # A = ?
    A=np.eye(6)+dt*np.eye(6,k=3)+np.diag([0, 0, 0, -c*dt, -c*dt, -c*dt])

    # a = ?
    a =  np.array([[0, 0, 0, 0, 0, g*dt]])
    
    # s = ?
    s = np.array([[0, 0, 2, 15, 3.5, 4.0]])
   
    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)

    S3=np.zeros([6,K])
    S3[:,0]=s

    for i in xrange(1,K):
        S3[:,i]=np.dot(A, S3[:,i-1])+a

    x3_coords=S3[0]
    y3_coords=S3[1]
    z3_coords=S3[2]

    ax.plot(x3_coords, y3_coords, z3_coords, '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    
    # B= ?
    B = np.diag([bx, by, bz, bvx, bvy, bvz])

    # C = ?
    C1=np.diag([rx,ry,rz])
    C2=np.zeros([3,3])
    C=np.bmat([[C1, C2]])

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)

    S4=np.zeros([6,K])
    S4[:,0]=s

    sk=s
    Sigmak=np.eye(6)

    for i in xrange(1,K):
        sk_tilde=np.dot(A,np.transpose(sk))+np.transpose(a)   
        Sigma_tilde=np.linalg.inv(np.dot(np.dot(A, Sigmak), np.transpose(A))+np.dot(B, np.transpose(B)))
        Sigmak=np.linalg.inv(Sigma_tilde+np.dot(np.transpose(C),C))
        xx=np.dot(Sigma_tilde, sk_tilde)+np.transpose(np.dot(np.transpose(C),np.transpose(s_m[i,:]))) # error: array+list
        sk=np.transpose(np.dot(Sigmak, xx))
        S4[:,i]=sk #error
   
    x4_coords=S4[0]
    y4_coords=S4[1]
    z4_coords=S4[2]
    
    ax.plot(x4_coords, y4_coords, z4_coords, '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
