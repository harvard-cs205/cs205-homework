import matplotlib.pyplot as plt
import numpy as np
import math

__author__ = 'Reinier Maat'

if __name__ == "__main__":

    n_bags = range(1, 65536, 1)

    def single_cashier(bags):
        return math.ceil(bags - 1)

    def infinite_cashiers(bags):
        return math.ceil(np.log2(bags))

    single_result = []
    for n in n_bags:
        single_result.append(single_cashier(n))

    infinite_result = []
    for n in n_bags:
        infinite_result.append(infinite_cashiers(n))

    plt.plot(n_bags, single_result, '-b')
    plt.plot(n_bags, infinite_result, '-g')
    plt.yscale('log')
    plt.xlabel('Bags')
    plt.ylabel('Time (seconds) Log-scale')
    plt.title('Time to count bags.')
    plt.show()

