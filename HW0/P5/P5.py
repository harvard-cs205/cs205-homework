import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1, 300)
y_inf = [np.ceil(np.log2(elt)) for elt in x]
y_1 = [elt - 1 for elt in x]

plt.plot(x, y_inf, '-b')
plt.plot(x, y_1, '-g')

plt.xlabel('Bag Count')
plt.ylabel('Counting Time')
plt.title('Count time for N bags - Infinite vs One employee(s)')
plt.legend(['Infinite Employees', 'Single Employee'], loc=1)
plt.savefig('P5.png') 
