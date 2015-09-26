# Your code here
import pyspark
sc = pyspark.SparkContext(appName="Spark1")

# make spark shut the hell up
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

import numpy as np 
import itertools

accum = sc.accumulator(0)

def remove_quotes(s):
	return s.replace('"','')

def get_pairs(arr):
	return list(itertools.permutations(arr, 2))

def reduceFun(tup):
	#global accum
	if tup[0] == None:
		accum.add(1)
		return tup[1]
	elif tup[1] == None:
		return tup[0]
	else:
		return min(tup[0], tup[1])

def reduceFun2(tup):
	if tup[1] == None:
		return tup[0]
	else:
		return max(tup[0],tup[1])

def bfs(adj, start):
	#global accum
	#return
	#print adj.lookup(start)
	accum.value = 0
	distances = adj.map(lambda (node, neighbor): (node, -1.0)).distinct()
	print distances.count()
	traversed = sc.parallelize([(start, 0)])
	farthest = 0
	accum.add(1)
	#i = 1
	#while farthest < 10:
	while accum.value != 0:
		accum.value = 0
		print "\n\nOn iteration ", farthest, ' for ', start, '\n\n'
		#if traversed.filter(lambda (node, dist): dist == farthest).count() == 0:
		#	break;
		# get all of the neighboring superheros
		#	Start by filtering for the ones that are farthest away
		# 	Then join this with the adjacency matrix, which gives you all of the places it can go
		#	Get just the neighbors by using .values()
		# 	Get rid of the distance value from the left side
		#	Keep only the unique ones using distinct (since there could be multipe ways to reach a node)
		#	Add the distance to them, using a lambda that makes the neighbors into KVs with Key = name, Value = dist
		neighbors = traversed.filter(lambda (node, dist): dist == farthest).join(adj).values().map(lambda x: x[1]).distinct().map(lambda x: (x, farthest + 1))
		# combine the neighbors with what we already had, removing values we have already seen.
		traversed = traversed.fullOuterJoin(neighbors).mapValues(reduceFun).distinct()
		#print traversed.keys().countByValue()
		# force the eval to calculate the accum values.
		print traversed.count()
		farthest += 1
		print 'Accum value: ', accum.value

	#print itera.take(10)
	final_vals = distances.leftOuterJoin(traversed).mapValues(reduceFun2)
	#print final_vals.take(100)
	print "Distance Distribution for ", start 
	print final_vals.values().countByValue(), '\n'
	
	#print "The ones that were unreachable:"
	#print final_vals.filter(lambda (name, dist): dist < 0).keys().collect()


wlist = sc.textFile('source.csv')
#Extract the strings and remove all unnessary quotes
better = wlist.map(lambda x: x.split('","')).map(lambda x: map(remove_quotes, x))
character_to_book = better.map(lambda x: (x[0], x[1]))
book_to_character = character_to_book.map(lambda (x, y): (y, x))
book_to_characters = book_to_character.groupByKey()

adjacent_heros = book_to_characters.flatMap(lambda (book, characters): get_pairs(characters)).distinct()
#print adjacent_heros.take(10)
#adjacency_matrix = adjacent_heros.groupByKey().mapValues(set).mapValues(list)
#adjacency_matrix.take(10)
#print adjacency_matrix.lookup(0)
bfs(adjacent_heros, 'CAPTAIN AMERICA')
bfs(adjacent_heros, 'MISS THING/MARY')
bfs(adjacent_heros, 'ORWELL')
#bfs(adjacent_heros, 'HOFFMAN')
#print adjacency_matrix.take(10)


