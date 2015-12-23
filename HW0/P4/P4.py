%matplotlib inline
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
    rvector=[rx, ry, rz]

    # Create 3D axes for plotting
    ax = Axes3D(plt.figure())
    
#####################
    # Part 1:
    # Load true trajectory and plot it
data = np.loadtxt('P4_trajectory.txt',delimiter=',')

print (len(data))

xcoords=[]
ycoords=[]
zcoords=[]
vxcoords=[]
vycoords=[]
vzcoords=[]


for i in range(len(data)):
    xcoords.append(data[i][0])
    ycoords.append(data[i][1])
    zcoords.append(data[i][2])
    vxcoords.append(data[i][3])
    vycoords.append(data[i][4])
    vzcoords.append(data[i][5])
    
# Normally, this data wouldn't be available in the real world
    #####################

ax.plot(xcoords, ycoords, zcoords,'-b', label='True trajectory')

#####################
    # Part 2:
data2 = np.loadtxt('P4_measurements.txt',delimiter=',')

# Read the observation array and plot it (Part 2)
xcoords2=[]
ycoords2=[]
zcoords2=[]


for i in range(len(data2)):
    xcoords2.append(data2[i][0]/rx)
    ycoords2.append(data2[i][1]/ry)
    zcoords2.append(data2[i][2]/rz)
    
ax.plot(xcoords2, ycoords2, zcoords2,'.g', label='Observed trajectory')  

#####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
#####################

a = np.zeros(shape=(1,6))
a[0][5]=g*dt
a=np.asmatrix(a)

    # Initial conditions for s0
s0 = [0, 0, 2, 15, 3.5, 4]
s0=np.asmatrix(s)
s0=np.transpose(s0)


# Compute the rest of sk using Eq (1)

A=np.zeros(shape=(6,6))
for i in range(3):
    A[i][i]=1
    
for i in range(3,6):
    A[i][i]=1-dt*c
    A[i-3][i]=dt
    
A=np.asmatrix(A)

sk=np.zeros(shape=(len(data),6))
sk=np.asmatrix(sk)


sk[0,:]=np.transpose(s0)
xcoords3=[]
ycoords3=[]
zcoords3=[]
for i in range((len(data)-1)):
    sk[i+1,:]= np.transpose(A*(np.transpose(sk[i,:]))) + a

sk=np.asarray(sk)

xcoords3=(sk[:,0])
ycoords3=(sk[:,1])
zcoords3=(sk[:,2])

ax.plot(xcoords3, ycoords3, zcoords3, '-r', label='Blind trajectory')


#####################
# Part 4:
# Use the Kalman filter for prediction
#####################
# Initial conditions for s0 are defined previously 
# Compute the rest of sk using Eqs (2), (3), (4), and (5)
def predictS(s):
    return np.transpose(A*s)+a

print predictS(s0)

B =np.zeros(shape=(6,6))
bvector=[bx, by, bz, bvx, bvy, bvz]
for i in range(6):
    B[i][i]=bvector[i]
    
B=np.asmatrix(B)

# Define Sigma0
covar=np.zeros(shape=(6,6))   ## This is sigma0

for i in range(6):
    covar[i][i]=1
    
covar=np.asmatrix(covar)


def predictSig(sigk):
    return np.linalg.inv(A*sigk*np.transpose(A)+ B*np.transpose(B))

C=np.zeros(shape=(3,6))

for i in range(3):
    C[i][i]=rvector[i]

C=np.asmatrix(C)

def updateSig(sigdash):
    return np.linalg.inv(sigdash+np.transpose(C)*(C))

# Turn data2 into a matrix called measurements to be able to use it for calculation
measmat=np.asmatrix(data2)
def updateS(sigk1,sigdash,sdash,mk1):
    return sigk1*(sigdash*np.transpose(sdash)+np.transpose(C)*mk1)

skk=np.zeros(shape=(len(data),6))
skk=np.asmatrix(skk)

skk[0,:]=np.transpose(s0)

for i in range(len(data)-1):
    sdash = predictS(np.transpose(skk[i,:]))
    sigdash = predictSig(covar)
    covar=updateSig(sigdash)
    skk[i+1,:]=np.transpose(updateS(covar,sigdash,sdash,np.transpose(measmat[i,:])))
 
skk=np.asarray(skk)
xcoords4=(skk[:,0])
ycoords4=(skk[:,1])
zcoords4=(skk[:,2])



ax.plot(xcoords4, ycoords4, zcoords4,'--k', label='Filtered trajectory')
ax.legend()
plt.show()

