import findspark
findspark.init()
import pyspark
import json
sc = pyspark.SparkContext(appName="")
from P4_bfs import bfs

## Import the character-comic pair.
source = sc.textFile('source.csv')

## For each comic (key), group the comic's characters into a list, then remove the comic key.
def split_raw(x):
    x1 = x.strip('"').split('","')
    if len(x1)==2:
        return (x1[1], x1[0])
source1 = source.map(split_raw).groupByKey().map(lambda x: list(x[1]))
## We are now left with lists of characters in each comic.

## In each list of characters, use each character as the key to create a list of character's neighbors.
## Then combine different neighbor lists for each character.
def comic_to_neighbor(vlist):
    neighbor = []
    for i in range(len(vlist)):
        neighbor.append((vlist[i], vlist[:i] + vlist[i+1:]))
    return neighbor
source2 = source1.flatMap(comic_to_neighbor).reduceByKey(lambda x,y: list(set(x+y))).map(lambda x: (x[0], (1000, x[1]))).cache()
## Now we have (k, v) for k=character and v=neighbors.

## Calling the bfs 
result1 = bfs(source2, 'CAPTAIN AMERICA')
result2 = bfs(source2, 'MISS THING/MARY')
result3 = bfs(source2, 'ORWELL')
