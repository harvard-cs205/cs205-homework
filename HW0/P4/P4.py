import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def predictS(A,s,a):
  return A*s+a

def predictSig(A,Sigma,B):
  return np.linalg.inv(A*Sigma*A.T + B*B.T)

def updateSig(Sigma,C):
  return np.linalg.inv(Sigma+C.T*C)

def updateS(Sigma_next,Sigma_appr,s,C,m):
  return Sigma_next*(Sigma_appr*s+C.T*m)

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

    traj_file = open("P4_trajectory.txt","r")
    s_true = np.loadtxt(traj_file, delimiter=",").T
    #print s_true.shape
    x = s_true[:3,:]
    v = s_true[3:,:]
    ax.plot(x[0,:], x[1,:], x[2,:],
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################

    measure_file = open("P4_measurements.txt","r")
    x_m = np.matrix(np.loadtxt(measure_file, delimiter=",").T)
    C_inv = np.matrix([[1/rx,0,0],[0,1/ry,0],[0,0,1/rz]])
    
    x_appr = C_inv * x_m
    x_coords = np.array(x_appr[0,:])[0]
    y_coords = np.array(x_appr[1,:])[0]
    z_coords = np.array(x_appr[2,:])[0]
    ax.plot(x_coords, y_coords, z_coords,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

    # Model parameters
    A = np.matrix([\
	[1,0,0,dt,0,0],\
	[0,1,0,0,dt,0],\
	[0,0,1,0,0,dt],\
	[0,0,0,1-c*dt,0,0],\
	[0,0,0,0,1-c*dt,0],\
	[0,0,0,0,0,1-c*dt]\
	])
    a = np.matrix([0,0,0,0,0,g*dt]).T
    s = np.matrix(np.zeros((6,K)))

    # Initial conditions for s0
    s0 = np.matrix([0,0,2,15,3.5,4.0]).T
    s[:,0] = s0
      
    # Compute the rest of sk using Eq (1)
    for i in range(K-1):
      s[:,i+1]= A*s[:,i]+a

    x_coords = np.array(s[0,:])[0]
    y_coords = np.array(s[1,:])[0]
    z_coords = np.array(s[2,:])[0]
    ax.plot(x_coords, y_coords, z_coords,
             '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    # For constructing matrix B
    bx = 0
    by = 0
    bz = 0
    bvx = 0.25
    bvy = 0.25
    bvz = 0.1
    B = np.matrix([\
	[bx,0,0,0,0,0],\
	[0,by,0,0,0,0],\
	[0,0,bz,0,0,0],\
	[0,0,0,bvx,0,0],\
	[0,0,0,0,bvy,0],\
	[0,0,0,0,0,bvz]\
	])
    C = np.matrix([\
	[rx,0,0,0,0,0],\
	[0,ry,0,0,0,0],\
	[0,0,rz,0,0,0],\
	])
    
    # Initial conditions for s0 and Sigma0
    Sigma = np.matrix(np.identity(6))
    Sigma = 0.01*Sigma
    
    # Initial conditions for s0
    s0 = np.matrix([0,0,2,15,3.5,4.0]).T
    s[:,0] = s0
    
    print type(x_m)
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    for i in range(K-1):
      # prediction steps
      s_appr = predictS(A,s[:,i],a)
      Sigma_appr = predictSig(A,Sigma,B)
      # update steps
      Sigma = updateSig(Sigma_appr,C)
      s[:,i+1] = updateS(Sigma,Sigma_appr,s_appr,C,x_m[:,i+1])

    x_coords = np.array(s[0,:])[0]
    y_coords = np.array(s[1,:])[0]
    z_coords = np.array(s[2,:])[0]

    ax.plot(x_coords, y_coords, z_coords,
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.savefig('P4.png')
    #plt.show()
