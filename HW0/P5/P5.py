import matplotlib.pyplot as plt
import math

counts = range(1, 500)

#For an infinite amount of employees to count a total with N bags,
#it takes log2(N) seconds.
infinite_employees_time = [math.ceil(math.log(n, 2)) for n in counts]

#For a single employee, it takes N+1 seconds to count the bags
single_employees_time = [n+1 for n in counts]

# plots data 
plt.plot(counts, single_employees_time, '-b', label="Single Employee")
plt.plot(counts, infinite_employees_time, '-g', label="Infinite Employees")
plt.xlabel('Bag Count')
plt.ylabel('Counting Time (seconds)')
plt.title('Count Speed by Algorithm')
plt.legend()
plt.show()