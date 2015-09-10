import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv

def predictS(A,s,a):
   	return (A*s+a)	
def predictSig(A,Sigma,B):
	return inv(A*Sigma*np.transpose(A)+B*np.transpose(B))
def updateSig(SigmaTilde,C):
	return inv(SigmaTilde+np.transpose(C)*C)
def updateS(Sigma,SigmaTilde,stilde,C,m):
	return Sigma*(SigmaTilde*stilde+np.transpose(C)*m)

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
    
    
    #####################
    ax.plot(s_true[:,0], s_true[:,1], s_true[:,2],'--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    s_meas=np.loadtxt('P4_measurements.txt',delimiter=',')
    cvec=np.array([rx,ry,rz])
    c1=np.asmatrix(np.diag(cvec))
    C=np.bmat([[c1, np.zeros([3,3])]])
    m=np.asmatrix(np.zeros([3,K]))

    s_t=np.bmat([[np.transpose(np.asmatrix(s_meas))],[np.asmatrix(np.ones([3,K]))]])
    print(C)
    for i in range(1,K):
    	m[:,i]=np.dot(C,s_t[:,i])
 #   m=np.bmat([np.bmat([s_meas[:,0]/rx,s_meas[:,1]/ry,s_meas[:,2]/rz]),
  #  		np.asmatrix(np.zeros([K,3]))]).reshape((6,K))

    #####################
    ax.plot(s_meas[:,0]/rx,s_meas[:,1]/ry,s_meas[:,2]/rz,
             '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    II=np.asmatrix(np.identity(3))
    A = np.bmat([[II, dt*II], [np.asmatrix(np.zeros([3,3])), (1-c*dt)*II]])
    #print(A)
    a = np.asmatrix(np.zeros([6,1]))
    a[5,0]=g*dt
    #print(a)
    s = np.asmatrix(np.zeros([6,K]))
	
    # Initial conditions for s0
    s[:,0]=np.transpose(np.matrix([[0,0,2,15,3.5,4.0]]))
    #print(s[:,0])
    for i in range(1,K):
    	s[:,i]=A*s[:,i-1]+a    	

    # Compute the rest of sk using Eq (1)
    xk=np.asarray(s)
    #print(xk[0,0:2])
    ax.plot(xk[0,:],xk[1,:],xk[2,:],
             '-k', label='Blind trajectory')

	#####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################
    #s should be 1d column vector
    
    bvec=np.array([bx,by,bz,bvx,bvy,bvz])	
    B = np.asmatrix(np.diag(bvec))
    
    # Initial conditions for s0 and Sigma0
    
    s[:,0]=np.transpose(np.matrix([[0,0,2,15,3.5,4.0]]))
    Sigma=0.01*np.asmatrix(np.identity(6))
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    for i in range (0,K-1):
		stilde=predictS(A,s[:,i],a)
		#print(stilde)
		#print(np.transpose(A))
		SigmaTilde=predictSig(A,Sigma,B)
		Sigma=updateSig(SigmaTilde,C)
		print(m[:,i])
		s[:,i+1]=updateS(Sigma,SigmaTilde,stilde,C,m[:,i])	
		
    ax.plot(s[1,:],s[2,:],s[3,:],
     	'-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
