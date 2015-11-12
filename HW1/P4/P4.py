"""
Written by Jaemin Cheun
Harvard CS205
Assignment 1
October 6, 2015
"""

import numpy as np
import findspark
findspark.init()
from P4_bfs import *

from pyspark import SparkContext

# initiaize Spark
sc = SparkContext("local", appName="P4")
sc.setLogLevel("ERROR")
marvel_list = sc.textFile("source.csv")

# make a key value pair where key is the issue and values are the character that are in the issue
def character_of_issue(line):
	line = line.split('"')
	return (line[3],line[1])

character_issue = marvel_list.map(character_of_issue)

# do a innner join to create tuples of characters that are in the same issue. Note that this also makes sure 
# that the links are symmetric, but it does not take care of link to itself. We then take the values because 
# we are no longer interested in the key/issue
edges = character_issue.join(character_issue).values()

# we first filter where character has a edge to itself, then group by key abd change the value to a list
edges = edges.filter(lambda (x,y) : x != y).groupByKey().mapValues(lambda x : list(x)).cache()

characterName="CAPTAIN AMERICA"
ss_bfs(edges,characterName,sc)

characterName="MISS THING/MARY"
ss_bfs(edges,characterName,sc)

characterName="ORWELL"
ss_bfs(edges,characterName,sc)


