import matplotlib.pyplot as plt
import numpy as np
N = np.power(2, 6)
def a(x):
    return np.log(x)
def b(x):
	return x - 1
x1 = np.arange(1,N,1)
plt.title('Time to add 64 bags.')
plt.plot(a(x1))
plt.plot(b(x1))
plt.ylabel('time in secods')
plt.xlabel('number of bags')
plt.text(40, 5, '$\eqslantgtr \/ N/2 \/ cashiers$')
plt.text(20, 30, '$1 \/ cashier$')
plt.show()