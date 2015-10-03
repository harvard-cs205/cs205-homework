import pyspark
from P4_bfs import *


def intersection(l1,l2):
	return bool(set(l1) & set(l2))

def getGraph_rev(data):
	'''Reverse map to (book, character) tuple and group by reduceByKey. O(N), used when characters are a lot.'''
	data = data.map(lambda l: l.split('"')).map(lambda l: [w for w in l if w!='' and w!=',']).map(lambda l: (l[0], l[1]))
	rev = data.map(lambda l: (l[1],l[0]))
	rev = rev.groupByKey().map(lambda l: list(l[1]))	# each entry is all characters in one book
	def remap(l):
	    res = []
	    for i in l:
	        temp1 = i
	        temp2 = l[:]
	        temp2.remove(i)
	        res += [(temp1, temp2)]
	    return res

	data2 = rev.flatMap(remap)	# has shape
	data3 = data2.groupByKey().map(lambda l: (l[0], list(l[1])))
	graph = data3.map(lambda l: (l[0], list(set([i for s in l[1] for i in s]))))

	return graph

def getGraph_cartesian(data):
	'''Use Cartesian method to return a transformed graph RDD of the form (chara, [neighbors])
	Complexity is O(#character^2)
	Better to use when characters are few while dataset is large.
	'''
	data = data.map(lambda l: l.split('"')).map(lambda l: [w for w in l if w!='' and w!=',']).map(lambda l: (l[0], l[1]))
	# dimension of data tested, all are two columns.

	# reduce to data2: list of (chara, [books])
	data2 = data.groupByKey().map(lambda l: (l[0], list(l[1])))

	# create graph: list of (chara, [neighbors])
	data_matrix = data2.cartesian(data2)	#creates n by n array for finding neighbors
	data3 = data_matrix.map(lambda l: (l[0][0], (l[0][1], l[1][0] if intersection(l[0][1],l[1][1]) and l[0][0]!=l[1][0] else '')))	#find neighbors
	data3 = data3.filter(lambda l: l[1][1]!='')	# clean up matrix
	data3 = data3.map(lambda l: (l[0], l[1][1]))
	#graph = data3.reduceByKey(lambda a,b: a+"%$%"+b).map(lambda l: (l[0], l[1].split("%$%")))
	graph = data3.groupByKey().map(lambda l: (l[0], list(l[1])))

	return graph



if __name__ == '__main__':
	sc = pyspark.SparkContext()
	sc.setLogLevel('WARN')

	# Read in file
	data = sc.textFile('source.csv')

	graph = getGraph_rev(data)

	result1 = BFS('CAPTAIN AMERICA', graph)
	result2 = BFS('MISS THING/MARY', graph)
	result3 = BFS('ORWELL', graph)

	print result1[:2]	#not printing all the nodes touched.
	print result2
	print result3