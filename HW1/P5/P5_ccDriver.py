from pyspark import SparkContext
import numpy as np
from itertools import chain
from P5_bfs import *
from connected_component import *

sc = SparkContext("local[4]",appName="ConnComp Testing")
sc.setLogLevel("ERROR")

#Read in the data
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
pages = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

#Creating the graph
assoc_pages = pages.zipWithIndex().mapValues(lambda v: v+1).partitionBy(32)

def mkInt(lst): return [int(val) for val in lst]
linksFlat =links.map(lambda s: s.split(": ")).map(lambda s: (int(s[0]),s[1])).mapValues(lambda l: l.split(" ")).mapValues(list).mapValues(lambda x: mkInt(x)).flatMapValues(lambda x: [t for t in x]).partitionBy(32)
revLinks = linksFlat.map(lambda x: (x[1], x[0]))

#Create symmetric graph of wikipedia links
symLinks = linksFlat.union(revLinks).map(lambda x: (x[0],[x[1]])).reduceByKey(lambda x, y: x+y).partitionBy(32).cache()

#Truncate graph to only those wikipedia links that are bidirectional
bidirLinks = linksFlat.intersection(revLinks).map(lambda x: (x[0],[x[1]])).reduceByKey(lambda x,y: x+y).partitionBy(32).cache()


connCOMPS_sym = connected_component(symLinks, sc)
connCOMPS_biDir = connected_component(bidirLinks, sc)

print "Number of components in Symmetric Graph: "+ len(connCOMPS_sym)
print "Maximum component in Symmetric Graph: " + max(connCOMPS_sym) + '\n\n'

print "Number of components in Bidirectional Graph: "+ len(connCOMPS_biDir)
print "Maximum component in Bidirectional Graph: "+ len(connCOMPS_biDir) + '\n\n'
