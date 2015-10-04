# (c) 2015 L.Spiegelberg
# provides code for Problem 4
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf

import csv
import pandas as pd
import os
import sys

import matplotlib.ticker as ticker   
import seaborn as sns
sns.set_style("whitegrid")


from P4_bfs import *

# convert Marvelgraph data to edge list using pandas
# this could be also done using SPARK, but in practice
# SPARK is really slow and this task needs to be done only once
def convertMarvelGraph2EdgeList(filename, edge_list_file, character_dict_file):
	comicdf = pd.read_csv(filename, names=['Character', 'ComicIssue'])
	keys = comicdf['Character'].unique()
	values = range(1, len(keys)+1) # start indices with 1

	characterDict = dict(zip(keys, values))

	# create graph by adjacency list (there is a connection between
	# two characters if they appear in the same comic issue)

	# therefore join the comicdf on comic issue!
	mergeddf = pd.merge(comicdf, comicdf, how='inner', on='ComicIssue')

	# now remove ComicIssue & all rows with Character_x == Character_y
	filtereddf = mergeddf.drop('ComicIssue', 1)
	filtereddf = filtereddf[filtereddf['Character_x'] != filtereddf['Character_y']]

	# transform string to integers for performance reasons
	edgedf = filtereddf.applymap(lambda x: characterDict[x])

	# save the edge list and the dictionary as two separate csv's
	edgedf.to_csv(edge_list_file, header=False, index=False)

	writer = csv.writer(open(character_dict_file, 'wb'))
	entries = sorted(characterDict.items(), key=lambda x: x[1]);
	for key, value in entries:
	    writer.writerow([value, key]) # flip it (so the vertex index is now the key)

# load vertex dictionary (is basically the character dictionary inverted)
def loadDictionaries(character_dict_file):
	reader = csv.reader(open(character_dict_file, 'rb'))
	vertexDict = dict(reader)

	characterDict = dict(zip(vertexDict.values(), [int(k) for k in vertexDict.keys()]))

	return vertexDict, characterDict


def loadEdgeList(sc, filename):
    rdd = sc.textFile(filename)

    # map string to tuples
    rdd = rdd.map(lambda x: x.split(','))
    rdd = rdd.map(lambda x: (int(x[0]), int(x[1])))
    
    # now group s.t. we have for each vertex an adjacency list of nodes
    rdd = rdd.groupByKey().map(lambda x: (x[0], list(x[1])))
    
    return rdd

# program logic
def main(argv):
	# setup spark
	conf = SparkConf().setAppName('Graph Processing')
	sc = SparkContext(conf=conf, pyFiles=['P4_bfs.py'])

	sc.setLogLevel('ERROR')

	filename = 'edge_list.csv'

	# create files lazy
	if not os.path.isfile('edge_list.csv'):
		convertMarvelGraph2EdgeList('source.csv', filename, 'characters.csv')

	# load dictionaries
	vertexDict, characterDict = loadDictionaries('characters.csv')

	# nodes to search for
	v0 = [characterDict['CAPTAIN AMERICA'], \
	   	  characterDict['MISS THING/MARY'], \
	      characterDict['ORWELL']]

	# write results to output.txt
	with open('output.txt', 'wb') as f:
		for i in range(0, len(v0)):
		    rdd = loadEdgeList(sc, filename)
		    num_visited_nodes, rdd = sparkBFS(sc, rdd, v0[i])
		    
		    message = '%s : %d nodes visited' % (vertexDict[str(v0[i])], num_visited_nodes)
		    print(message)
		    f.write(message + '\n')


# avoid to run code if file is imported
if __name__ == '__main__':
	main(sys.argv)