from P4_bfssat import * 
import pyspark
from pyspark import SparkContext
sc = SparkContext("local[4]")

sc.setLogLevel("ERROR")

import numpy as np
import matplotlib.pyplot as plt 
data = sc.textFile('data.txt')
Superheros = data.map(lambda word: word.split('",')[0]).map(lambda word: word.strip('"').encode("utf-8"))
Comics= data.map(lambda word: word.split('",')[1]).map(lambda word: word.strip('"').encode("utf-8"))

RDD1= Comics.zip(Superheros)
RDD2 = RDD1.join(RDD1) #K,(V,W) = Comic, (Superhero,Superhero)

#create edge list and ensure no one is linked to themselves (this assumes
#every character has at least one neighbor otherwise if they are solo then
#they will never be found as they are omitted from this graph)...
Graph = RDD2.filter(lambda x: x[1][0]!=x[1][1]).map(lambda x: (x[1][0],x[1][1])).distinct().partitionBy(8).cache()

SOURCE = 'ORWELL'
SOURCE2 = 'MISS THING/MARY'
SOURCE3 = 'CAPTAIN AMERICA'

count1=bfs(SOURCE,Graph,sc)
count2=bfs(SOURCE2,Graph,sc)
count3=bfs(SOURCE3,Graph,sc)

print count1
print count2
print count3