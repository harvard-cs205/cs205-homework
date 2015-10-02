import findspark
findspark.init()
import pyspark
import itertools
from P4 import *


if __name__ == "__main__":  
    sc = pyspark.SparkContext()
    
    print source().lookup("LANN")
