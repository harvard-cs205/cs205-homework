import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
        
    # Tells us how many seconds it would take for an inf number of cashiers to complete the sum
    def inf_cashier(x):            
        return np.ceil(np.log2(n)) # rounds up the base-2 logarithm of the input

    min = 1
        # Minimum number of bags that we are plotting
    max = 68
        # Maximum number of bags that we are plotting
    n = range(min, max + 1)
        # The range of inputs for the infinite cashiers function

    inf = inf_cashier(n)       # The number of seconds it takes for infinite cashiers
    one = range(min - 1, max)  # The number of seconds it takes for one cashier

    plt.plot(one, '-b', label = 'Single Cashier')
    plt.plot(inf, '-r', label = 'Infinite Cashiers')
    plt.legend(loc = 'best')
    plt.xlabel('Number of bags')
    plt.ylabel('Number of seconds')
    plt.title('Amount of time it takes cashiers to sum bags')
    plt.show()




    
