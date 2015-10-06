# Your code here
import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark1")

from P4_bfs import *

# Reduce the Spark Logs
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

import numpy as np 
import itertools

def remove_quotes(s):
    return s.replace('"','')

def get_pairs(arr):
    return list(itertools.permutations(arr, 2))

num_Partitions = 16

#Extract the strings and remove all unnessary quotes
wlist = sc.textFile('source.csv', num_Partitions)
character_to_book = wlist.map(lambda x: x.split('","')).map(lambda x: map(remove_quotes, x))

#Switch the order of the map to be from book to characters
book_to_character = character_to_book.map(lambda (x, y): (y, x))
book_to_characters = book_to_character.groupByKey(numPartitions = num_Partitions)

# Extract all pairs of characters that share a book.
adjacent_heros = book_to_characters.flatMap(lambda (book, characters): get_pairs(characters)).distinct().partitionBy(num_Partitions)

# Create the graph as an adjacency list.
graph = adjacent_heros.groupByKey().partitionBy(num_Partitions).cache()

res1 = bfs(graph, 'CAPTAIN AMERICA', sc, num_Partitions)
print "Distance Breakdown from CAPTAIN AMERICA (-1 indicates unreachable)"
print res1.values().countByValue(), '\n'

res1 = bfs(graph, 'MISS THING/MARY', sc, num_Partitions)
print "Distance Breakdown from MISS THING/MARY (-1 indicates unreachable)"
print res1.values().countByValue(), '\n'

res1 = bfs(graph, 'ORWELL', sc, num_Partitions)
print "Distance Breakdown from ORWELL (-1 indicates unreachable)"
print res1.values().countByValue(), '\n'
