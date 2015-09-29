from pyspark import SparkContext
import numpy

sc=SparkContext()
titlesTxt = sc.textFile('small_titles-sorted.txt')
txt = sc.textFile('small_links-simple-sorted.txt')

indexByTitle = titlesTxt.zipWithIndex().map(lambda t: (t[0],t[1]+1))

startNode = '1'
def titleToId(title):
	return str(indexTitle.lookup(title)[0])
#(id,distance,[id,id,...]), initialize starting node's distance to 0
graph = txt.map(lambda s:(s.split(' ')[0][:-1],10000, tuple(s.split(' ')[1:])  )  ).map(lambda s:(s[0],0,s[2]) if s[0]==startNode else s)
def extraArgs(neighbors,distance):
	def updateNode(n):
		if n[0] in neighbors and n[1] > distance+1:
			return (n[0],distance+1,n[2])
		else:
			return n
	return updateNode
		
greyNodes= set(graph.filter(lambda x:x[0]==startNode).collect())

for i in range(10):
	for x in greyNodes.copy():
		neighbors = x[2]
		distance = x[1]
		greyNodes.remove(x)
		graph = graph.map(extraArgs(neighbors,distance))
		greyNodes.update(graph.filter(lambda y:y[0] in neighbors).collect())
			
endNode = '5'
print graph.filter(lambda y:y[0]==endNode).collect()