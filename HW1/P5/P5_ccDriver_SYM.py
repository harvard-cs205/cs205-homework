from pyspark import SparkContext
import numpy as np
from itertools import chain
from P5_connected_component import *

sc = SparkContext()
sc.setLogLevel("ERROR")

#Read in the data
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',32)

def mkInt(lst): return [int(val) for val in lst]
linksFlat =links.map(lambda s: s.split(": ")).map(lambda s: (int(s[0]),s[1])).mapValues(lambda l: l.split(" ")).mapValues(list).mapValues(lambda x: mkInt(x)).flatMapValues(lambda x: [t for t in x])
revLinks = linksFlat.map(lambda x: (x[1], x[0])).cache()

#Create symmetric graph of wikipedia links
symLinks = linksFlat.union(revLinks).groupByKey().mapValues(list).partitionBy(256).cache()


connCOMPS_sym = P5_connected_component(symLinks, sc)

print "Number of components in Symmetric Graph: "+ len(connCOMPS_sym)
print "Maximum component in Symmetric Graph: " + max(connCOMPS_sym) + '\n\n'


