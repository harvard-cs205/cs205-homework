import numpy as np
import matplotlib.pyplot as plt

xaxis = np.array([ii for ii in range(2,10**3)])
plt.plot(xaxis,np.ceil(np.log2(xaxis)),'b',label='parallel (infinite people)')
plt.plot(xaxis,xaxis-1,'r',label='serial (1 person)')
plt.legend()
plt.xlabel('Number of bags')
plt.ylabel('Time to Count bags')
plt.title('Parallel and Serial Count Times vs Number of Bags')
plt.show()
