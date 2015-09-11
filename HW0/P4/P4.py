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
    x_coords,y_coords,z_coords,vx,vy,vz = np.loadtxt('P4_trajectory.txt',
        delimiter=',',unpack=True)
    
    ax.plot(x_coords, y_coords, z_coords,
             '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    x_obs,y_obs,z_obs = np.loadtxt('P4_measurements.txt',
        delimiter=',',unpack=True)
    # include stretch
    ax.plot(x_obs/rx, y_obs/ry, z_obs/rz,
         '.g', label='Observed trajectory')

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    A = np.matrix([[1.,0.,0.,dt,0.,0.],[0.,1.,0.,0.,dt,0.],
        [0.,0.,1.,0.,0.,dt],[0.,0.,0.,1-c*dt,0.,0.],
        [0.,0.,0.,0.,1-c*dt,0.],[0.,0.,0.,0.,0.,1-c*dt]])
    a = np.matrix([[0,0,0,0,0,g*dt]]).T
    s = np.zeros((6,120))
    for i in xrange(len(x_coords)):
        s[:,i] = np.matrix([[x_coords[i],y_coords[i],z_coords[i],vx[i],vy[i],vz[i]]])
    s = np.matrix(s).T
    
    # Initial conditions for s0
    s0 = np.matrix([[0.,0.,2.,15.,3.5,4.0]]).T
    s[0] = s0.T

    # Compute the rest of sk using Eq (1)
    for i in xrange(len(x_coords)-1):
        s[i+1] = np.matrix(A*np.matrix(s[i]).T+a).T 
    
    x_sim = np.zeros(len(x_coords))
    y_sim = np.zeros(len(x_coords))
    z_sim = np.zeros(len(x_coords))
    for i in xrange(len(x_coords)):
        x_sim[i] = s[i,0]
        y_sim[i] = s[i,1]
        z_sim[i] = s[i,2]
    
    ax.plot(x_sim, y_sim, z_sim,'-k', label='Blind trajectory')
   
    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.diag([bx,by,bz,bvx,bvy,bvz])
    # stretch in measurements
    C = np.concatenate((np.diag([rx,ry,rz]),np.diag([0,0,0])),axis=1)
    #print C
    # measurement matrix
    m = np.zeros((3,120))
    for i in xrange(len(x_coords)):
        m[:,i] = np.matrix([[x_obs[i],y_obs[i],z_obs[i]]])
    m = np.matrix(m).T
    # Initial conditions for s0 and Sigma0
    sig0 = 0.01*np.identity(6)
    
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    def predictS(s_k):
        s = A*s_k+a
        return s
    
    def predictSig(sig_k):
        f = A*sig_k*A.T + np.dot(B,B.T)
        return np.linalg.inv(f)
    
    def updateSig(sig):
        f = sig + np.dot(C.T,C)
        return np.linalg.inv(f)
    
    def updateS(sig_k,sig_t,s,m):
        ss = sig_k*(sig_t*s + C.T*m)
        return ss
    
    K = len(x_coords)
    x_kf = np.zeros(K)
    y_kf = np.zeros(K)
    z_kf = np.zeros(K)
    sig_temp = sig0
    
    for i in xrange(K-1):
        s_pred = predictS(s[i].T)
        sig_pred = predictSig(sig_temp)
        sig_new = updateSig(sig_pred)
        #print updateS(sig_new,sig_pred,s_pred,m[i+1].T).T
        s[i+1] = updateS(sig_new,sig_pred,s_pred,m[i+1].T).T
        sig_temp = sig_new
    for i in xrange(K):
        x_kf[i] = s[i,0]
        y_kf[i] = s[i,1]
        z_kf[i] = s[i,2]  
    ax.plot(x_kf, y_kf, z_kf,
             '-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
