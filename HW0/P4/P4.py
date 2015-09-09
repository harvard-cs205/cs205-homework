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

read_temp=np.loadtxt('P4_trajectory.txt',delimiter=',')
xx=(np.transpose(read_temp))[0]
yy=(np.transpose(read_temp))[1]
zz=(np.transpose(read_temp))[2]
    #####################

ax.plot(xx, yy, zz,'--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
read_temp_meas=np.loadtxt('P4_measurements.txt',delimiter=',')
C=np.matrix([[1/rx,0,0],[0,1/ry,0],[0,0,1/rz]])
sm=np.transpose(read_temp_meas)
xk=C*sm
x_apx=np.squeeze(np.asarray(xk[0]))
y_apx=np.squeeze(np.asarray(xk[1]))
z_apx=np.squeeze(np.asarray(xk[2])) 
	#####################

ax.plot(x_apx, y_apx, z_apx,'.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
s0=np.matrix([[0],[0],[2],[15],[3.5],[4.0]])
A=np.matrix([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1-c*dt,0,0],[0,0,0,0,1-c*dt,0],[0,0,0,0,0,1-c*dt]])
a=np.matrix([[0],[0],[0],[0],[0],[g*dt]])
s=np.zeros((120,6))
    
    # s = ?


    # calculate each position
s_temp=s0
s[0]=np.transpose(s0)
i=1
while i < 120:
	s_temp1=A*s_temp+a
	s[i]=np.transpose(s_temp1)
	s_temp=s_temp1
	i+=1


    # Initial conditions for s0
    # Compute the rest of sk using Eq (1)
s_transpose=np.transpose(s)
x_cal=s_transpose[0]
y_cal=s_transpose[1]
z_cal=s_transpose[2]
ax.plot(x_cal, y_cal, z_cal,'-k', label='Blind trajectory')
	
    #####################
    # Part 4:
    # Use the Kalman filter for prediction
def predictS(s):
	A=np.matrix([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1-c*dt,0,0],[0,0,0,0,1-c*dt,0],[0,0,0,0,0,1-c*dt]])
	a=np.matrix([[0],[0],[0],[0],[0],[g*dt]])
	s_bar=A*s+a
	return s_bar

def predictSig(sigma):
	A=np.matrix([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1-c*dt,0,0],[0,0,0,0,1-c*dt,0],[0,0,0,0,0,1-c*dt]])
	B=np.matrix([[bx,0,0,0,0,0],[0,by,0,0,0,0],[0,0,bz,0,0,0],[0,0,0,bvx,0,0],[0,0,0,0,bvy,0],[0,0,0,0,0,bvz]])
	sigma_bar=(A*sigma*(np.transpose(A))+B*(np.transpose(B))).I
	return sigma_bar

def updateSig(sigma_bar):
	C=np.matrix([[rx,0,0,0,0,0],[0,ry,0,0,0,0],[0,0,rz,0,0,0]])
	sigma_kplus=(sigma_bar+(np.transpose(C))*C).I
	return sigma_kplus

def updateS(s_bar,sigma_bar,sigma_kplus,m):
	C=np.matrix([[rx,0,0,0,0,0],[0,ry,0,0,0,0],[0,0,rz,0,0,0]])
	mm=np.asmatrix(m)
	s_kplus=sigma_kplus*((sigma_bar*s_bar)+(np.transpose(C))*(np.transpose(mm)))
	return s_kplus


    # Initial conditions for s0 and Sigma0
s0=np.matrix([[0],[0],[2],[15],[3.5],[4.0]])
sigma_temp=0.01*(np.eye(6))
ss=np.zeros([121,6])
ss[0]=np.transpose(s0)
s_temp=s0
i=1
# should use read_temp_meas[i]) or read_temp_meas[i-1])
while i < 121:
	s_bar=predictS(s_temp)
	sigma_bar=predictSig(sigma_temp)
	sigma_kplus=updateSig(sigma_bar)
	s_kplus=updateS(s_bar,sigma_bar,sigma_kplus,read_temp_meas[i])
	ss[i]=np.transpose(s_kplus)
#update the temporary variable s and sigma
	s_temp=s_kplus
	sigma_temp=sigma_kplus
	i+=1


    #####################

# Compute the rest of sk using Eqs (2), (3), (4), and (5)
ss_transpose=np.transpose(ss)
x_KF=ss_transpose[0]
y_KF=ss_transpose[1]
z_KF=ss_transpose[2]
ax.plot(x_KF, y_KF, z_KF,'-r', label='Filtered trajectory')

# Show the plot
ax.legend()
plt.show()

