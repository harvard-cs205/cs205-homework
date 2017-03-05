import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def parallelTime(N):
    powerTwo = 1 << (N.bit_length() - 1)
    if N > powerTwo:
        powerTwo = powerTwo << 1
    return powerTwo.bit_length() - 1

def serialTime(N):
    return N - 1

if __name__ == '__main__':
    sampleN_Base = [2, 5, 10]
    tens = 1

    Ns, parallelTs, serialTs = [], [], []
    for i in xrange(3):
        for n in sampleN_Base:
            sampleN = n * tens
            Ns.append(sampleN)
            parallelTs.append(parallelTime(sampleN))
            serialTs.append(serialTime(sampleN))

        tens *= 10

    plt.plot(Ns, parallelTs, '-r', label='Parallel Times')
    plt.plot(Ns, serialTs, '-b', label='SeriaParallell Times')

    # Show the plot
    plt.show()
