from pyspark import SparkContext
import numpy


sc=SparkContext()

#titlesTxt = sc.textFile('titles-sorted.txt')
txt = sc.textFile('links-simple-sorted.txt')

#indexByTitle = titlesTxt.zipWithIndex().map(lambda t: (t[0],t[1]+1))

startNode = '2729536'
def titleToId(title):
	return str(indexTitle.lookup(title)[0])
#(id,distance,[id,id,...]), initialize starting node's distance to 0
graph = txt.map(lambda s:(s.split(' ')[0][:-1],10000, tuple(s.split(' ')[1:]),False  )  ).map(lambda s:(s[0],0,s[2],True) if s[0]==startNode else s)
# graph = txt.map(lambda s:(s.split(' ')[0][:-1],(10000, tuple(s.split(' ')[1:]),False)  )  ).map(lambda s:(s[0],(0,s[1][1],True)) if s[0]==startNode else s).partitionBy(10)
# print graph.take(1)
# greyNodes= set(graph.filter(lambda x:x[0]==startNode).collect())

# def extraArgs(neighbors,distance):
# 	def updateNode(n):
# 		if n[0] in neighbors and n[1] > distance+1:
# 			if n[3] == False:
# 				return (n[0],distance+1,n[2],True)
# 			else:
# 				return (n[0],distance+1,n[2],n[3])
# 		else:
# 			return n
# 	return updateNode

# def f(neighbors):
# 	return lambda y:y[3]==False and y[0] in neighbors

# for i in range(3):
# 	neighbors = set([])
# 	print len(greyNodes)
# 	for x in greyNodes.copy():
# 		neighbors.update(x[2])
# 		greyNodes.remove(x)
# 	greyNodes.update(graph.filter(f(neighbors)).collect())
# 	graph = graph.map(extraArgs(neighbors,i))
# 	graph.count()


#DO NOT UNCOMMENT
# graph = txt.map(lambda s:(s.split(' ')[0][:-1],(10000, tuple(s.split(' ')[1:]),False)  )  ).map(lambda s:(s[0],(0,s[1][1],True)) if s[0]==startNode else s)
# greyNodes= graph.filter(lambda x:x[0]==startNode)
# for i in range(3):
# 	print 'Grey Count =',greyNodes.count()
# 	neighbors = greyNodes.flatMap(lambda x:x[1][1]).map(lambda x:(x,0))
# 	greyNodes = neighbors.leftOuterJoin(graph).map(lambda x:(x[0],x[1][1]))
	# greyNodes =  graph.filter(f(neighbors))
	# graph = graph.map(extraArgs(neighbors,i))
	# graph.count()
#DO NOT UNCOMMENT


# endNode = '2152852'
# print graph.filter(lambda y:y[0]==endNode).collect()


