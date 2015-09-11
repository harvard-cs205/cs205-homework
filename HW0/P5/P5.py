# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 05:38:35 2015

@author: sdeppen
"""
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    

    x_coords = np.arange(0,100,1)
    
    
    y_coords = np.log2(x_coords)
    plt.plot(x_coords, y_coords, '-r', label='infinite')
    plt.plot(x_coords, x_coords, '-g', label='lone')
    
    # Show the plot
    plt.xlabel('number of bags')
    plt.ylabel('seconds')
    plt.legend()
    plt.show()