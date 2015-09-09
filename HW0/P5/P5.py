import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
N = [i for i in range(256,500)]
dt = 1
time_take = []
time_take_2 = []
for i in N:
    j = 0
    time_take.append(i-1)
    while True:
		if i/2 != 0:
			i = i/2
			j+=1
			continue
		time_take_2.append(j)
		break
print time_take,time_take_2
plt.plot(N,time_take_2,'b',N,time_take,'r')
plt.xlabel('Time steps')
plt.ylabel('Number of bags')
plt.suptitle('P5 image')
plt.savefig('P5.png')

plt.show()

