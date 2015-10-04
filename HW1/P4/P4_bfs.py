#import findspark
#findspark.init()
#import pyspark
#import itertools
#from P4 import *


# as I understand it, this file should be just a function that is called in P4.py

def BFS(sc, graph, name):
    
    # Rdd of name, shortest distance tuples that will be returned.  In each iteration, heros are added to this.  Initialize with the source node, and partition into 4 partitions to match the graph.
    rdd = sc.parallelize([(name, 0)], 4)
    
    # This RDD is used intermediately to build a collection of connections to build on
    namesRdd = rdd

    # Iterator used in the while loop, capped at 10
    i = 0
    
    while (i < 10) and not namesRdd.isEmpty(): # Terminate if nothing was added in the last iteration
    
        i = i+1    
    
        # Look up the connections to the last added and take a flat map of just the connections
        #namesRdd = graph.filter(lambda (x,y): x in names).flatMap(lambda (x,y): y)
 
        # Use join to look up the connections of the next round of connections
        namesRdd = graph.join(namesRdd)

        # Use mapValues to take just the list of connections and remove the distance
        namesRdd = namesRdd.mapValues(lambda (x,y): x)

        # Take the connections and flatMap them into one big pool to consider
        namesRdd = namesRdd.values().flatMap(lambda x: x)

        # Reduce the connections to distinct entries
        namesRdd = namesRdd.distinct()
        
        # Subtract out the names that are already in the rdd
        namesRdd = namesRdd.subtract(rdd.keys())

        # create tuples of the connection name and the current distance of the iteration
        namesRdd = namesRdd.map(lambda x: (x, i)).partitionBy(4)  

        # Union the connection tuples to the existing list, and repartition
        rdd = rdd.union(namesRdd).partitionBy(4)
    
    return rdd
        
