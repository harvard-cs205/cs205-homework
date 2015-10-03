# Your code here
import pyspark
sc = pyspark.SparkContext(appName="Spark1")
from P4_bfs import *

# make spark shut the hell up
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

import numpy as np 
import itertools

def remove_quotes(s):
	return s.replace('"','')

def get_pairs(arr):
	return list(itertools.permutations(arr, 2))

numPartitions = 15

wlist = sc.textFile('source.csv', numPartitions)
#Extract the strings and remove all unnessary quotes
better = wlist.map(lambda x: x.split('","'), True).map(lambda x: map(remove_quotes, x), True)
character_to_book = better.map(lambda x: (x[0], x[1]))
book_to_character = character_to_book.map(lambda (x, y): (y, x))
book_to_characters = book_to_character.groupByKey()

adjacent_heros = book_to_characters.flatMap(lambda (book, characters): get_pairs(characters)).partitionBy(numPartitions).distinct()

res1, res2 = bfs(adjacent_heros, 'CAPTAIN AMERICA', sc, numPartitions)
print res1.values().countByValue(), '\n'
#bfs(adjacent_heros, 'MISS THING/MARY', sc)
#bfs(adjacent_heros, 'ORWELL', sc)
#bfs(adjacent_heros, 'HOFFMAN')




#print adjacency_matrix.take(10)


