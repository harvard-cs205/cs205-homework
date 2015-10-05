###################################################
### Problem 4 - Graph processing in Spark [25%] ###
### Patrick Day 								###
### CS 205 HW1                                  ###
### Oct 4th, 2015								###
###################################################

########################
### Import Functions ###
########################
from __future__ import division
from __future__ import print_function

import pyspark
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import timeit
import csv
import string
import itertools
from random import shuffle
from operator import add

import os
sc = pyspark.SparkContext()
sc.setLogLevel('WARN')

########################
### Define Functions ###
########################
def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

### Import Source ###
p4_path = "/Users/pday/Dropbox/Harvard/Fall15/CS205/HW1/source.csv"
lines = []
with open(p4_path, 'rb') as csvfile:
    source_csv = csv.reader(csvfile)
    for row in source_csv:
        lines.append(row)

#########################
### Build Heros Graph ###
#########################
char_rdd = sc.parallelize(lines).partitionBy(100)
char_rdd_kv = char_rdd.map(lambda x: (x[1], x[0]))

char_rdd_kv = char_rdd.map(lambda x: (x[1], x[0]))
char_group_rdd = char_rdd_kv.groupByKey()
char_list = char_group_rdd.mapValues(list).collect()
char_list_1 = sc.parallelize(char_list).partitionBy(100)
char_rdd_2 = char_list_1.flatMap(lambda x: list(itertools.combinations(x[1], 2)))
#print(char_list_1.take(5), '\n')
#print(char_rdd_2.take(10), '\n')

char_rdd_reverse = char_rdd_2.map(lambda x: (x[1], x[0]))
union = char_rdd_2.union(char_rdd_reverse)
union_2 = union.groupByKey().mapValues(list)
#union_3 = union_2.map(lambda x: (x[0], list(set(x[1]))))#.collectAsMap()


########################
### Initialize Parent ##
########################
graph_init = union_2.map(lambda x: (x[0], (x[1], 100, 0))).partitionBy(100)
hero1 = 'CAPTAIN AMERICA'

# Create Hero RDD
hero_values = graph_init.filter(lambda x: x[0] == hero1)#.flatMap(lambda x: x)

# Set Parent to Processing
parent = hero_values.map(lambda x: (x[0][0], (x[1][0], 0, 1)), preservesPartitioning=True)

# Get Neighbors and update to processed
neighbor = parent.collect()[0][1][0]
nei_rdd = graph_init.filter(lambda x: x[0] in neighbor)
nei_rdd = nei_rdd.map(lambda x: (x[0], (x[1][0], 1, x[1][2]+1)), preservesPartitioning=True) 
#print("Nei", nei_rdd.collect(), "\n")

# Check if neighbor and final graph are copartitioned
print("Neighbor and final graph copart:", copartitioned(nei_rdd, graph_init))

# Add back to final graph
graph_init = graph_init.union(nei_rdd).reduceByKey(lambda x,y: min(x,y))

# Update Parent
parent_update = parent.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2]+1)), 
                           preservesPartitioning=True)

# Check Partition to ensure that they are the same size
print("Parent and final graph copart:", copartitioned(parent_update, graph_init))

# Add updated parent back to final graph and cache
graph = graph_init.union(parent_update).reduceByKey(lambda x,y: min(x, y)).cache()

##########################
### Execute BFS Search ###
##########################

parent_count = 10
depth = 2
while (depth < 6 and parent_count!=0):
    
    # Turn child into Parent
    parent = graph.filter(lambda x: x[1][-1] == 1)
    parent_count = parent.count()
    print("Parents:", parent_count)
    
    # Get Neighbors
    neighbor = set(parent.map(lambda x: x[1][0], preservesPartitioning=True).flatMap(lambda x: x).collect())
    #print("Nei RDD", graph.filter(lambda x: x[0] in neighbor).collect())
    
    # Get Neighbors, Make sure no parents are present 
    # And update to processed
    nei_rdd = graph.filter(lambda x: x[0] in neighbor).filter(lambda x: x[1][-1] < 2)
    
    # Changed to process and add to graph
    nei_rdd = nei_rdd.map(lambda x: (x[0], (x[1][0], depth, 1)), preservesPartitioning=True)
    #print ("Neighbor processed", nei_rdd.collect())
    graph = graph.union(nei_rdd).reduceByKey(lambda x,y: min(x,y))
                                                       
    # Update Parent
    parent_update = parent.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2]+1)), preservesPartitioning=True)
    #print("Parents Updated:", parent_update.collect(),"\n")
    graph = graph.union(parent_update).reduceByKey(lambda x,y: max(x, y)).cache()
    
    depth+=1

print(graph.collect()[-1])

