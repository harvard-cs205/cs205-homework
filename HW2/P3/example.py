import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
from mandelbrot import example_sqrt_8

test = np.array([3 + 4j, 3 + 4j, 3 + 4j, 3 + 4j, 3 + 4j, 3 + 4j, 3 + 4j, 3 + 4j])

print example_sqrt_8(test.astype(np.complex64))
