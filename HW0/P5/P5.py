import numpy as np
import matplotlib.pyplot as plt

bags = np.arange(2,50,1)
logTime = np.log2(bags)
plt.plot(bags,logTime,'.g', label='Infinite Workers')
linTime = bags - 1
plt.plot(bags,linTime,'.b', label='One Worker')
plt.ylabel('Time')
plt.xlabel('Bags')
plt.legend()
plt.show()