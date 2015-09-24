from P4_bfs import *
from pyspark import SparkContext, SparkConf

#Setup
conf = SparkConf().setAppName("comic_graph").setMaster("local")
sc = SparkContext(conf=conf)

#Generate mapping from character to other characters they appear with
#Take the cross-product of characters who appear in a given issue with itself, reduceByKey to form a set of all characters who appear with each character, remove key character from set
source = sc.textFile("source.csv")
issue_character = source.map(lambda line: (line.split(',')[1], line.split(',')[0]))
character_characters = issue_character.join(issue_character).map(lambda KV: (KV[1][0], {KV[1][1]} - {KV[1][0]})).reduceByKey(lambda x, y: x.union(y))

#BFS
origins = ['"CAPTAIN AMERICA"', '"MISS THING/MARY"', '"ORWELL"']
results = {}
for origin in origins:
    result = bfs(character_characters, origin)
    results[origin] =  result.count()
print results
