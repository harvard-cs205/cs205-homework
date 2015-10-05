
# coding: utf-8

# In[1]:

import findspark
import os
findspark.init('/home/chongmo/spark') # you need that before import pyspark.
import pyspark
from pyspark import SparkContext
sc =SparkContext()


# In[4]:

import ast
import itertools


# In[5]:

#convert the initial RDD (name, issue) into a RDD (issue, {all characters in the issue})
series_RDD = sc.textFile('source.csv').map(lambda x: ast.literal_eval(x.encode('utf-8').decode('ascii', 'ignore'))[::-1]).groupByKey().mapValues(lambda x: set(x))


# In[6]:

#convert series_RDD into a RDD (node, {neighbor1, neighbor2, ...})
neighbors_RDD=series_RDD.flatMap(lambda x: ((i, tuple(x[1].difference(set([i])))) for i in x[1])).reduceByKey(lambda x, y: x+y).mapValues(lambda x: set(x))


# In[27]:

def SS_BFS(G, node_source): #G:graph RDD, (char1, {char2, char3, ...})    
    #distance=sc.parallelize([(node_source, 0)])
    queue = sc.parallelize([node_source])
    visited=sc.parallelize([node_source])
    #d=0
    #while d<diameter/2:
    while not queue.isEmpty(): 
        q=set(queue.collect())
        v=set(visited.collect())
        neighbors=G.filter(lambda x: x[0] in q).flatMap(lambda x: (x[1])).distinct().filter(lambda x: not(x in v))
        if not neighbors.isEmpty():
            queue=neighbors
            #d=d+1
            #distance=distance.union(neighbors.map(lambda x: (x, d)))
            visited=visited.union(neighbors)     
        else:
            break   
    return visited.count() 


# In[12]:

test1=SS_BFS(neighbors_RDD, 'CAPTAIN AMERICA')


# In[30]:

test2=SS_BFS(neighbors_RDD, 'MISS THING/MARY')


# In[28]:

test3=SS_BFS(neighbors_RDD, 'ORWELL')

