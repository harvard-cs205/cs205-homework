from __future__ import division
from P4_bfs import *
import numpy as np
import pyspark

sc = pyspark.SparkContext(appName = "Spark1")

marvelData = sc.textFile('marvel.csv').cache()

marvelList = marvelData.map(lambda x: x.split('"'))
charL = marvelList.map(lambda x: (x[1],x[3])).partitionBy(10)
comicL = marvelList.map(lambda x: (x[3],x[1])).partitionBy(10)

characterFirst = charL.groupByKey().mapValues(list)
characterList = characterFirst.map(lambda x: x[0]).cache()
comicFirst = comicL.groupByKey().mapValues(list).map(lambda x: (x[0],np.unique(x[1]))).partitionBy(10)

charAdj = comicL.join(comicFirst).map(lambda x: (x[1][0],x[1][1])).groupByKey().mapValues(list)
charAdj = charAdj.map(lambda x: (x[0],np.array([item for sublist in x[1] for item in sublist])))
charAdj = charAdj.map(lambda x: (x[0],np.unique(x[1][x[1] != x[0]]))).partitionBy(10).cache() # make adjacency list

#startName = 'CAPTAIN AMERICA'
#startName = 'MISS THING/MARY'
startName = 'ORWELL'  # root of BFS tree here

componentSize = bfsSearch(startName, charAdj, sc)

print componentSize # print list of connected nodes at each BFS iteration

