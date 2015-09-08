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
    
    C = np.diag([rx,ry,rz])

    # Create 3D axes for plotting
    ax = Axes3D(plt.figure())

    #####################
    # Part 1:
    #
    # Load true trajectory and plot it
    # Normally, this data wouldn't be available in the real world
    #####################
    
    TruTraj = np.loadtxt("P4_trajectory.txt",dtype='float',delimiter=',')
    xTru_coords, yTru_coords, zTru_coords = TruTraj[:,0], TruTraj[:,1], TruTraj[:,2]

    #Plotting the true position
    ax.plot(xTru_coords, yTru_coords, zTru_coords, '--b', label='True trajectory')

    #####################
    # Part 2:
    #
    # Read the observation array and plot it (Part 2)
    #####################
    
    ObsTraj = np.loadtxt("P4_measurements.txt",dtype='float',delimiter=',')
    xObs_coords, yObs_coords, zObs_coords = ObsTraj[:,0], ObsTraj[:,1], ObsTraj[:,2]
    
    #Plotting the measured position
    #ax.plot(xObs_coords, yObs_coords, zObs_coords,'.g', label='Observed trajectory')
    
    xtld = ObsTraj.dot(np.diag([1/rx, 1/ry, 1/rz])) # I decided to try and streamline this...
                                           #  which may be a sign that I'm getting comfy
    #Plotting the approximate position
    ax.plot(xtld[:,0],xtld[:,1],xtld[:,2],'.r', label='Approximate Position')                                        
    
    

    #####################
    # Part 3:
    # Use the initial conditions and propagation matrix for prediction
    #####################

	#Propagation Matrix
    A = np.diag([1, 1, 1, 1-c*dt, 1-c*dt, 1-c*dt]) + np.diag([dt, dt, dt],3)
    
    #Accounting for gravity in update
    a = np.zeros((A.shape[0])) 
    a[-1] = g*dt
    a = a.reshape([6,1])
    # s = ?

    # Initial conditions for s0
    s0 = np.matrix(TruTraj[0,:]).T
    
    # Compute the rest of sk using Eq (1)
    sK = np.matrix(np.zeros((6,np.amax(TruTraj.shape)))) #Initializing array for state predictions
    sK[:,0] = s0                              #Placing initial measurement as first column
    
    for i in range(1,np.amax(TruTraj.shape)):
         sK[:,i] = A * sK[:,i-1] + a
    
    sK = np.asarray(sK) # Converting state prediction matrix to an array (for plotting??)
    
    #Plotting the predicted positions
    ax.plot(sK[0,:], sK[1,:], sK[2,:], '-k', label='Blind trajectory')

    #####################
    # Part 4:
    # Use the Kalman filter for prediction
    #####################

    B = np.diag([bx, by, bz, bvx, bvy, bvz])
    C = np.concatenate( ( np.diag([rx, ry, rz]), np.zeros((3,3)) ), axis=1 )
    
    # Initial conditions for s0 and Sigma0
    s0_kalman, Sigma0 = np.matrix(TruTraj[0,:]).T, np.identity(A.shape[0]) * 0.01
    
    # Compute the rest of sk using Eqs (2), (3), (4), and (5)
    
    def predictS(priorState): # Eq(2)
        #Will fill out code structure here based on state update from part 3
        predS = np.add(np.dot(A,priorState),a)
        return predS
         
    def predictSig(priorSig): # Eq(3)
        #Predict SigmaK update
        predSig = np.linalg.inv(np.add(np.dot(np.dot(A, priorSig),np.transpose(A)),np.dot(B,np.transpose(B))))
        return predSig
         
    def updateSig(predSig): # Eq(4)
        #Update SigmaK
        sigUpdate = np.linalg.inv(np.add(predSig,np.dot(np.transpose(C),C)))
        return sigUpdate
         
    def updateS(predS,sigmaUpdate,predSig,curMeas): # Eq (5)
        #Update state estimate
        stateUpdate = np.dot(sigmaUpdate, np.add(np.dot(predSig,predS), np.dot(np.transpose(C),curMeas).reshape([6,1])))
        return stateUpdate
         
    # Set up state estimation data structure and convert measurements array to a matrix
    sK_kalman = np.matrix(np.zeros((6,np.amax(TruTraj.shape))))
    sK_kalman[:,0] = s0_kalman
    Measurements = ObsTraj
    
         
     # Run Kalman filter to determine estimate of trajectory     
    for ii in range(1,np.amax(TruTraj.shape)):
        if ii == 1:
            sigUpdate = Sigma0
              
        sPred = predictS(sK_kalman[:,ii-1])
        sigPred = predictSig(sigUpdate)
        sigUpdate = updateSig(sigPred)
        sK_kalman[:,ii] = updateS(sPred,sigUpdate,sigPred,Measurements[ii,:].T)
    
    
    sK_kalman = np.asarray(sK_kalman) # Converting state prediction matrix to an array (for plotting??)

    ax.plot(sK_kalman[0,:], sK_kalman[1,:], sK_kalman[2,:],'-r', label='Filtered trajectory')

    # Show the plot
    ax.legend()
    plt.show()
