import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
from mandelbrot import example_sqrt_8

print example_sqrt_8(np.arange(8, dtype=np.float32))
