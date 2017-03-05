import matplotlib.pyplot as plt
import numpy as np
import math

# Main
if __name__ == '__main__':
    
    n_bags = range(1,100)
    infinite_cashiers = [ math.ceil(math.log(x)) for x in n_bags ]
    lone_cashier = [x - 1 for x in n_bags ]
    print(n_bags)
    print(infinite_cashiers)
    print(lone_cashier)


    # Plot the results
    plt.plot(n_bags, infinite_cashiers, '-r', label='Infinite Cashiers')
    plt.plot(n_bags, lone_cashier, '-b', label='Lone Cashier')
    plt.legend(loc=2)
    plt.xlabel('N bags')
    plt.ylabel('Total time (seconds)')
    plt.title('Total time vs. N bags')
    plt.show()