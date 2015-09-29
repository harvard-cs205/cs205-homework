from pyspark import SparkContext
import networkx as nx
import Queue
import pdb

sc=SparkContext()

def nonRecursiveBFS(Graph,startingNode,endingNode):
	q = Queue.Queue()
	
	Graph.node[startingNode]['distance'] = 0
	q.put(startingNode)
	shortestPath=-1
	while not q.empty():
		node = q.get()
		if node == endingNode: 
			shortestPath = Graph.node[node]['distance']
			break
		for c in Graph.neighbors(node):
			if Graph.node[c]['distance'] == -1:
				Graph.node[c]['distance'] = Graph.node[node]['distance']+1
				q.put(c)
	return shortestPath

def resetDistance(Graph):
	for n in Graph.nodes():
		Graph.node[n]['distance']=-1

def idToTitle(id,titles):
	return titles.lookup(id)
def titleToId(title,indexByTitle):
	return indexByTitle.lookup(title)

# txt = sc.textFile('small_links-simple-sorted.txt')
txt = sc.textFile('small_links-simple-sorted.txt',100)
#List of tuples ('fromid',['to1','to2',...])
txt2 = txt.map(lambda s:(s.split(' ')[0][:-1],s.split(' ')[1:]))
edges=txt2.flatMap(lambda t:[(t[0],x) for x in t[1]])

#titlesTxt = sc.textFile('small_titles-sorted.txt')
titlesTxt = sc.textFile('small_titles-sorted.txt')
indexByTitle = titlesTxt.zipWithIndex().map(lambda t: (t[0],t[1]+1)).cache()

#Node names are string ids
totalNodes = titlesTxt.count()
nodeNames = [str(n) for n in range(1,totalNodes+1)]#starting from index 1

G = nx.DiGraph()
for n in nodeNames:
	G.add_node(n,distance=-1)
# javaIterator = edges._jrdd.toLocalIterator()
# edgeIterator = edges._collect_iterator_through_file(javaIterator)
# for edge in edgeIterator:
# 	G.add_edge(edge[0],edge[1])

# def make_part_filter(index):
#     def part_filter(split_index, iterator):
#         if split_index == index:
#             for el in iterator:
#                 yield el
#     return part_filter

# for part_id in range(edges.getNumPartitions()):
#     partialEdgesPartition = edges.mapPartitionsWithIndex(make_part_filter(part_id), True)
#     edgesFromPartition = partialEdgesPartition.collect()
#     for e in edgesFromPartition:
#     	G.add_edge(e[0],e[1])

totalOverNumParts = totalNodes/5
for i in range(30):
	toAddEdges = edges.filter(lambda t: i*totalOverNumParts < int(t[0]) <= (i+1)*totalOverNumParts).collect()
	for e in toAddEdges:
		G.add_edge(e[0],e[1])

# resetDistance(G)
# print nonRecursiveBFS(G,
# 	str(titleToId('Kevin_Bacon',indexByTitle)[0]),
# 	str(titleToId('Harvard_University',indexByTitle)[0]))








