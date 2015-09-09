import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
N = [2**i for i in range(1,25)]
dt = 1
time_take = []
for i in N:
    j = 0
    while True:
		if i/2 != 0:
			i = i/2
			j+=1
			continue
		time_take.append(j)
		break
plt.plot(N,time_take,'', label='aha')
plt.xlabel('Time steps')
plt.ylabel('Number of bags')
fig.suptitle('P5 image')
fig.savefig('P5.png')

#plt.show()

