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
s_true=np.loadtxt("P4_trajectory.txt", delimiter=",",unpack=True)

ax.plot(s_true[0,:], s_true[1,:], s_true[2,:],
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
m=np.loadtxt("P4_measurements.txt", delimiter=",",unpack=True)
notc=np.diag([1/rx,1/ry,1/rz])
xbar= np.asarray(notc * np.asmatrix(m))

ax.plot(xbar[0,:], xbar[1,:], xbar[2,:],
           '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

A = np.zeros(shape=(6,6))
A[0,0]=A[1,1]=A[2,2]=1.
A[0,3]=A[1,4]=A[2,5]=dt
A[3,3]=A[4,4]=A[5,5]=1.-c*dt

a = np.array([0.,0.,0.,0.,0.,g*dt])
s = np.array([0.,0.,2.,15.,3.5,4.])

S=np.zeros(shape=(6,K))
S[:,0]=s
S=np.asmatrix(S)
    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
for i in range(0,(K-1)):
	S[:,(i+1)]=np.asarray((np.asmatrix(A)*np.asmatrix(S[:,i]))+np.asmatrix(a).T)
S2=np.asarray(S)
ax.plot(S2[0], S2[1], S2[2],
            '-k', label='Blind trajectory')

  #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

B = np.zeros(shape=(6,6))
B[0,0]=bx
B[1,1]=by
B[2,2]=bz
B[3,3]=bvx
B[4,4]=bvy
B[5,5]=bvz
B=np.asmatrix(B)

C = np.zeros(shape=(6,6))
C[0,0]=rx
C[1,1]=ry
C[2,2]=rz
C=np.asmatrix(C)

    # Initial conditions for s0 and Sigma0
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
Sigma=Sigma0=np.identity(6)*.1
s0=s

Skalman=np.zeros(shape=(6,K))
Skalman[:,1]=s0
Skalman=np.asmatrix(Skalman)
mt=np.append(m,np.zeros(shape=(3,121)),0)

def predictS(A,Skalman,i,a):
	return np.asarray((np.asmatrix(A)*np.asmatrix(Skalman[:,i]))+np.asmatrix(a).T)

def predictSig(Sigma,A,B):
	return np.linalg.inv(A*Sigma*A.T+B*B.T)

def updateSig(Sigmatil,C):
	return np.linalg.inv(Sigmatil+C.T*C)

def updateS(Sigma,Sigmatil,stil,C,m,i):
	return np.asarray(Sigma*(Sigmatil*stil+C.T*np.asmatrix(m[:,i+1]).T))

for i in range(0,(K-1)):
	stil = predictS(A,Skalman,i,a)
	Sigmatil=predictSig(Sigma,A,B)
	Sigma=updateSig(Sigmatil,C)
	Skalman[:,(i+1)]=updateS(Sigma,Sigmatil,stil,C,mt,(i))

#for i in range(0,(K-1)):
#	stil=np.asarray((np.asmatrix(A)*np.asmatrix(Skalman[:,i]))+np.asmatrix(a).T)
#	Sigmatil=np.linalg.inv(A*Sigma*A.T+B*B.T)
#	Sigma=np.linalg.inv(Sigmatil+C.T*C)
#	Skalman[:,(i+1)]=np.asarray(Sigma*(Sigmatil*stil+C.T*np.asmatrix(mt[:,(i+1)]).T))

Skalman2=np.asarray(Skalman)
ax.plot(Skalman2[0], Skalman2[1], Skalman2[2],
            '-r', label='Filtered trajectory')  

    # Show the plot
ax.legend()
plt.show()
