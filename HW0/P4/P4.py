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
    
    traj=np.loadtxt("P4_trajectory.txt", delimiter=",");
    np.reshape(traj,(120,6));
    traj=np.transpose(traj)
    x_coords,y_coords,z_coords=traj[0:3,:];

    ax.plot(x_coords, y_coords, z_coords,
            '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    
    meas=np.loadtxt("P4_measurements.txt", delimiter=",");
    np.size(meas)
    np.reshape(meas,(K,3));
    meas=np.transpose(meas)

    invr_vec = np.array([1./rx,1./ry,1./rz])
    invC= np.diag(invr_vec)
    m=np.dot(invC,meas)

    x_coords,y_coords,z_coords=m[0:3,:];
    ax.plot(x_coords, y_coords, z_coords, '.g', label='Observed trajectory')
    
    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################
    
    A = np.eye(6)+np.diag(dt*np.ones(3),3)-np.diag(np.concatenate((np.zeros(3), c*dt*np.ones(3))))
    a = np.concatenate((np.zeros(5),np.array([g*dt])))
    s = []


    # Initial conditions for s0
    s0= np.array([0., 0., 2., 15., 3.5, 4.0])
    s.append(s0)
    
    # Compute the rest of sk using Eq (1)
    for k in range(1,K,1):
        sk = np.dot(A,s[-1])+a
        s.append(sk)
    s=np.array(s)
    np.reshape(s,(6,K))
    s=np.transpose(s)
    
    x_coords,y_coords,z_coords=s[0:3,:]
    ax.plot(x_coords, y_coords, z_coords,
            '-k', label='Blind trajectory')
    
    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.diag(np.array([bx,by,bz,bvx,bvy,bvz]))
    C = np.zeros((3,6))
    C[:3,:3] = np.diag(np.array([rx,ry,rz]))
    
    def prefictS(s):
        return np.dot(A,s)+a
    
    def predictSig(sig):
        return np.linalg.inv(np.dot(np.dot(A,sig),np.transpose(A))+np.dot(B,np.transpose(B)))
    
    def updateSig(sig):
        return np.linalg.inv(sig+np.dot(np.transpose(C),C))
    
    def updateS(sigk,sig,s,m):
        return np.dot(sigk,(np.dot(sig,s)+np.dot(np.transpose(C),m)))


    # Initial conditions for s0 and Sigma0
    s2 = []
    s0= np.array([0., 0., 2., 15., 3.5, 4.0])
    s2.append(s0)
    Sigma0=np.eye(6)*.01
    
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    curs = prefictS(s0)
    curSig = predictSig(Sigma0)
    curSigk = updateSig(curSig)
    
    for k in range(1,K,1):
        s2.append(updateS(curSigk,curSig,curs,meas[:,k]))
        curs = prefictS(s2[-1])
        curSig = predictSig(curSigk)
        curSigk = updateSig(curSig)
    s2=np.array(s2)
    np.reshape(s2,(6,K))
    s2=np.transpose(s2)

    x_coords, y_coords, z_coords = s2[0:3,:]
    ax.plot(x_coords, y_coords, z_coords,
            '-r', label='Filtered trajectory')
    
    # Show the plot
    ax.legend()
    plt.show()
