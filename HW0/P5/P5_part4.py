import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set_context('poster', font_scale=1.25)

# Note that we made the xscale start at N=2, or else it doesn't make sense

N = np.logspace(np.log10(2), 6, num=500)
t_alone = N - 1
t_infinite = np.log2(N)
plt.loglog(N, t_alone, label=r'One Cashier, $\Delta t = (N - 1)t_{add}$')
plt.loglog(N, t_infinite, label=r'Infinite Cashiers, $\Delta t = \log_2[N]t_{add}$')

plt.legend(loc='best')

plt.xlabel(r'$N$ Bags')
plt.ylabel(r'Time (s)')
plt.title(r'Time Required to Count $N$ Bags')
plt.savefig('P5.png', dpi=200, bbox_inches='tight')