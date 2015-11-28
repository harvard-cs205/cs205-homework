import matplotlib.pyplot as plt
import math
import numpy as np

x = np.linspace(0.5,300,1000);

y1 = [math.log(i,2) for i in x] 
y2 = [i-1 for i in x] 

p1, = plt.plot(x,y1,'b');
p2, = plt.plot(x,y2,'r');
plt.ylim([0,10]);
plt.ylabel("Counting time(sec)");
plt.xlabel("Bag counts");
plt.legend([p1,p2],["Multiple people","One person"]);
plt.title("HW1 Part 4");
plt.show();
