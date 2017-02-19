import numpy as np
import matplotlib.pyplot as plt

n=50
s=np.zeros(n)
s_one=np.zeros(n)
i=2
s[0]=0
s[1]=0
s_one[0]=0
s_one[1]=0
while i < n:
	s_one[i]=i-1
	for j in range(0,i):
		if i <= 2**j:
			pass
		elif i <= 2**(j+1):
			s[i]=j+1
		else:
			continue


	i+=1

# END
print i
x=range(0,n)
plt.plot(x,s,'.b',label='unlimited cashiers')
plt.plot(x,s_one,'.k',label='one cashier')
plt.xlabel('number of bags')
plt.ylabel('time(s)')
legend=plt.legend(loc='upper left')
plt.show()
			

