
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%matplotlib inline


# In[2]:

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


# In[3]:

plot_params = []

def get_plot_params(data,lty,lcol,lab):
    x = np.squeeze(np.asarray(data[:,0]))
    y = np.squeeze(np.asarray(data[:,1]))
    z = np.squeeze(np.asarray(data[:,2]))
    return (x, y, z, lty, lcol, lab)


# In[4]:

#####################
# Part 1:
#
# Load true trajectory and plot it
# Normally, this data wouldn't be available in the real world
#####################

# load trajectory data
s_true = np.loadtxt("P4_trajectory.txt", delimiter=",")

plot_params.append( get_plot_params(s_true, "--", "b", "True trajectory") )


# In[5]:

#####################
# Part 2:
#
# Read the observation array and plot it (Part 2)
#####################

# load observation data
M = np.loadtxt("P4_measurements.txt", delimiter=",").T

# C matrix, defined in HW0.P4.2
Cleft  = np.diag([rx,ry,rz])
Cright = np.zeros((3,3))
C      = np.bmat( 'Cleft Cright' )

# ~X^k, defined in HW0.P4.2
Xpred = np.dot( np.diag([1/rx,1/ry,1/rz]), M )

plot_params.append( get_plot_params(Xpred.T, ".", "g", "Observed trajectory") )


# In[6]:

#####################
# Part 3:
# Use the initial conditions and propagation matrix for prediction
#####################

# assemble A matrix quadrants
UL = np.identity(3)
LL = np.zeros((3,3))
UR = np.identity(3)*dt
LR = np.identity(3)*(1-c*dt)

# define A matrix from quadrants
A  = np.bmat('UL UR; LL LR')

a  = np.append(np.zeros(5),g*dt)
S  = np.zeros((6,K))

# Initial conditions for s0
s0 = s_true[0,:]

# Compute the rest of sk using Eq (1)
s = s0
for k in xrange(K):
    s = np.squeeze(np.asarray((np.dot( A, s.T ) + a.T)))
    S[:,k] = s

plot_params.append( get_plot_params(S.T, "*", "r", "Predicted trajectory") )


# In[7]:

#####################
# Part 4:
# Use the Kalman filter for prediction
#####################

C = np.bmat( 'Cleft Cright' )
B = np.identity(6)*[bx,by,bz,bvx,bvy,bvz]
M = np.matrix(M)
a = np.append(np.zeros(5),g*dt)
S = np.zeros((6,K))

# Initial conditions
s0     = s_true[0,:]
s_k    = np.matrix(s0).T
sig0   = np.identity(6)*0.01
sig_k  = sig0
S[:,0] = s0

# Compute the rest of sk using Eqs (2), (3), (4), and (5)
def predictS(A,s,a):
    return np.dot( A, s ) + np.matrix(a).T

def predictSig(sigk):
    return np.linalg.inv( np.dot(A, np.dot(sigk,A.T) ) + np.dot(B,B.T))
    
def updateSig(sigpr):
    return np.linalg.inv( sigpr + np.dot(C.T,C) )
    
def updateS(sigk,sigpr,spr,mnext):
    return np.dot( sigk, (np.dot(sigpr, spr) + np.dot(C.T,mnext)) )

for k in xrange(K-1):
    m_next   = M[:,k+1]
    s_pred   = predictS(A,s_k,a)
    sig_pred = predictSig(sig_k)
    sig_k    = updateSig(sig_pred)
    s_k      = updateS(sig_k, sig_pred, s_pred, m_next)
    S[:,k+1] = np.squeeze(np.asarray(s_k))

plot_params.append( get_plot_params(S.T, "*", "c", "Kalman trajectory") )


# In[8]:

## plot trajectories

fig = plt.figure(figsize=(10,8))
ax  = fig.add_subplot(111, projection='3d')

for params in plot_params:
    x_coords = params[0]
    y_coords = params[1]
    z_coords = params[2]
    lty = params[3]
    lcol = params[4]
    lab  = params[5]
    
    ax.plot(x_coords, y_coords, z_coords, lty+lcol, label=lab)

ax.azim=70
ax.elev=30
_=ax.legend() 


# In[ ]:



