import matplotlib.pyplot as plt

__author__ = 'Reinier Maat'

if __name__ == "__main__":

    n_bags = range(1, 65536, 256)

    # Single case, all is added to the first pile
    def time_taken_for_number_of_cashiers(number_of_bags, number_of_cashiers):
        cashiers_in_use = 0
        if number_of_cashiers > number_of_bags:
            cashiers_in_use = number_of_bags    # other cashiers have nothing to do
        else:
            cashiers_in_use = number_of_cashiers
        return (number_of_bags/cashiers_in_use) + (cashiers_in_use - 1)  # communication to main cashier

    single_cashier = 1
    single_result = []
    for n in n_bags:
        single_result.append(time_taken_for_number_of_cashiers(n, single_cashier))

    infinite_cashiers = 1e99
    infinite_result = []
    for n in n_bags:
        infinite_result.append(time_taken_for_number_of_cashiers(n, infinite_cashiers))

    plt.plot(n_bags, single_result, '-b')
    plt.plot(n_bags, infinite_result, '--g')
    plt.xlabel('bags')
    plt.ylabel('time')
    plt.title('Time to count bags. Cashiers in infinite case have been limited to the number of bags')
    plt.show()

