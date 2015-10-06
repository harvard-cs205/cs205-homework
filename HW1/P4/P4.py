
import pyspark
from pyspark import SparkContext
#from P4_bfs import BFS, iterate
from P4_bfs2 import BFS, iterate
import time

sc = SparkContext()
#f = open('graph.txt', 'w')
chars = open('chars.txt', 'w')


def char_to_edges(lstc):
	lst = lstc[1]
	dic = {}
	all_edges = []
	for charA in lst:
		dic[charA] = []
		for charB in lst:
			if charA != charB:
				dic[charA].append(charB)
		all_edges.append((charA, dic[charA]))
	return all_edges

rawData = sc.textFile('source.csv')

# separate = rawData.map(lambda x: x.split("\",\""))
# noQuotes = separate.map(lambda x: (x[0].replace("\"", ''), x[1].replace("\"", '')))

separate = rawData.map(lambda x: (x.split('"')[1], x.split('"')[3]))


#print noQuotes.take(1)

reverse_and_listed = separate.map(lambda x: (x[1], [x[0]]))

comic_chars = reverse_and_listed.reduceByKey(lambda x, y: x + y)



#comic_chars.map(lambda x, y: char_to_edges(y))	
graph_edges = comic_chars.flatMap(char_to_edges)


c = graph_edges.reduceByKey(lambda x, y: x+y)


d = c.map(lambda (x, y): (x, list(set(y))))



#print >> f, d.collect()

#print a.take(2)

print "count of Captain America Touches : ", BFS(sc, d, 'CAPTAIN AMERICA').filter(lambda (x, y): True if y[1] <  1000 else False).count()
print "count of Miss Thing Touches : ", BFS(sc, d, 'MISS THING/MARY').filter(lambda (x, y): True if y[1] <  1000 else False).count()
print "count of Orwell Touches : ", BFS(sc, d, 'ORWELL').filter(lambda (x, y): True if y[1] <  1000 else False).count()



