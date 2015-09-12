import math
import matplotlib.pyplot as plt

# Main
if __name__ == '__main__':

    # Calculate number of seconds for 5-500 bags
    bags = [x for x in range(5,505,5)]
    
    # Time required for lone cashier = N-1
    lone_cashier = [x-1 for x in bags]
    
    # Time required for infinite cashiers = log-2(N)
    inf_cashiers = [math.log(x,2) for x in bags]

    # Plot the results
    # Use subplots b/c lines are on very different scales
    plot, axis1 = plt.subplots()
    axis2 = axis1.twinx()
    
    axis1.plot(bags, lone_cashier, '--b', label='Lone cashier')
    axis2.plot(bags, inf_cashiers, '-r', label='Infinite cashiers')
    
    axis1.set_xlabel('Number of bags')
    axis1.set_ylabel(ylabel='Lone cashier time (seconds)', color='blue')
    axis2.set_ylabel(ylabel='Infinite cashier time (seconds)', color='red')
    
    plt.show()