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


    # Part 1:
    s_true=np.loadtxt('P4_trajectory.txt',delimiter=',')
    true_x=s_true[:,0]
    true_y=s_true[:,1]
    true_z=s_true[:,2]

    ax.plot(true_x, true_y, true_z,'--b', label='True trajectory')

    
    # Part 2:
    s_measure=np.loadtxt('P4_measurements.txt',delimiter=',')
    measure_x=s_measure[:,0]/rx
    measure_y=s_measure[:,1]/ry
    measure_z=s_measure[:,2]/rz
    
    ax.plot(measure_x, measure_y, measure_z,'.g', label='Observed trajectory')

    # Part 3:
    A=np.array([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1-c*dt,0,0],[0,0,0,0,1-c*dt,0],[0,0,0,0,0,1-c*dt]])
    a=np.array([[0],[0],[0],[0],[0],[g*dt]])
    S=np.zeros((6,K))

    # Initial conditions for s0
    S[:,0]=list([0,0,2,15,3.5,4])

    # Compute the rest of sk using Eq (1)
    i=0
    while i<K-1:
        S[:,i+1]=(np.dot(A,S[:,i].reshape((6,1)))+a).reshape(6)
        i+=1

    px=S[0,:].reshape(K)
    py=S[1,:].reshape(K)
    pz=S[2,:].reshape(K)
   
    ax.plot(px,py,pz,'-k', label='Blind trajectory')


    # Part 4:
    B=np.diag((bx,by,bz,bvx,bvy,bvz))
    C=np.concatenate((np.diag((rx,ry,rz)),np.zeros((3,3))))

    # Initial conditions for s0 and Sigma0
    Sigma=0.01*np.matrix(np.identity(6))
    s=np.zeros((6,K))
    s[:,0]=list([0,0,2,15,3.5,4])

    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    j=0
    while j<K-1:
    	s_predict=np.dot(A,s[:,j].reshape((6,1)))+a
        Sigma_predict=np.linalg.inv(np.dot(np.dot(A,Sigma),np.transpose(A))+np.dot(B,np.transpose(B)))
	Sigma=np.linalg.inv(Sigma_predict+np.dot(C,np.transpose(C)))
	s[:,j+1]=(np.dot(Sigma,(np.dot(Sigma_predict,s_predict.reshape((6,1)))+np.dot(C,s_measure[j+1,:].reshape((3,1)))))).reshape(6)
    	j+=1

    qx=s[0,:].reshape(K)
    qy=s[1,:].reshape(K)
    qz=s[2,:].reshape(K)
    
    ax.plot(qx,qy,qz,'-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
