import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
sns.set_context('poster', font_scale=1.25)
from random import randint

# INITIALIZING PYSPARK
import pyspark as ps

config = ps.SparkConf()
config = config.setMaster('local[' + str(2*mp.cpu_count()) + ']')
config = config.setAppName('P6')

sc = ps.SparkContext(conf=config)
# REMOVE ALL OF THE DIFFERENT SPARK WARNINGS
sc.setLogLevel('WARN')

