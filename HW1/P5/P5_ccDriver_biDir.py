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


#Truncate graph to only those wikipedia links that are bidirectional
bidirLinks = linksFlat.intersection(revLinks).groupByKey().mapValues(list).partitionBy(256).cache()


#Run connComp on both derived graphs
connCOMPS_biDir = P5_connected_component(bidirLinks, sc)

print "Number of components in Bidirectional Graph: "+ len(connCOMPS_biDir)
print "Maximum component in Bidirectional Graph: "+ max(connCOMPS_biDir) + '\n\n'



