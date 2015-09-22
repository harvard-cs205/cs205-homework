from pyspark import SparkContext

import networkx as nx
import matplotlib.pyplot as plt
import itertools
import pdb
import Queue

sc = SparkContext()
textFile = sc.textFile('source.csv')
#Convert to |Hero Name|,|Comic| then take out the Comic and place on left, Hero on right. Later we will groupBy() using comic as key
formatedTf = textFile.map(
	lambda x:(x.replace('"','|').split('|')[3].strip(),x.replace('"','|').split('|')[1].strip()))
#We group by Comic, and the value of each key(comic) will be an array of heros 
#in that comic
groupedTf = formatedTf.groupByKey().map(lambda t:(t[0],list(t[1])))
#Using N choose K method from itertools, I take pairs of heros from the same
#comic, then flatten the tuples array. This data will be used as edges later on
#I sort beforehand so we can eliminate scenarios such as (A,B),(B,A). We only want one of them
heroRelationTuplesNoDup = groupedTf.flatMap(lambda t: list(itertools.combinations(sorted(t[1]),2)))
#Remove duplicates by putting list in set
heroTuplesNoDup = set(heroRelationTuplesNoDup.collect())
#Take just the hero names from text file

uniqueHeros = textFile.map(lambda x:x.replace('"','|').split('|')[1].strip())
print uniqueHeros.take(100)
#Take the unique names
uniqueHeroNames = set(uniqueHeros.collect())


#Draws nodes
G = nx.Graph()
#add distance attr to each node
for x in uniqueHeroNames:
	G.add_node(x,distance=-1)
#Add edges
for edge in heroTuplesNoDup:
	G.add_edge(edge[0],edge[1])
# print G.neighbors('CAPTAIN AMERICA')
# plt.figure(figsize=(20,20))
# nx.draw_random(G,with_labels=False,node_size=1,width=0.1)
# plt.savefig('graph.png')
# print '24-HOUR MAN/EMMANUEL' in G.nodes()
# print 'IRON MAN/TONY STARK' in uniqueHeroNames
# print 'IRON MAN/TONY STARK' in uniqueHeros.collect()

def nonRecursiveBFS(Graph,startingNode):
	q = Queue.Queue()
	Graph.node[startingNode]['distance'] = 0
	q.put(startingNode)
	currentMax=0
	while not q.empty():
		node = q.get()
		nodeDist = Graph.node[node]['distance']
		print currentMax,nodeDist
		currentMax = nodeDist if nodeDist > currentMax else currentMax
		for c in Graph.neighbors(node):
			if Graph.node[c]['distance'] == -1:
				Graph.node[c]['distance'] = Graph.node[node]['distance']+1
				q.put(c)
	return currentMax
def shortestPath(Graph,startingNode,count):
	pdb.set_trace()
	children = []
	distances=[Graph.node[startingNode]['distance']]
	# if len(Graph.successors(startingNode)) == 0:
	# 	return Graph.node[startingNode]['distance']

	for c in Graph.neighbors(startingNode):
		if Graph.node[c]['distance'] == -1:
			Graph.node[c]['distance'] = count+1
			distances.append(shortestPath(Graph,c,count+1))
	
	return max(distances)
	#if no children, then return current node's distance
#Reset distance of each node to 0
def resetDistance(Graph):
	for n in Graph.nodes():
		Graph.node[n]['distance']=-1

resetDistance(G)
startingNode='CAPTAIN AMERICA'
print nonRecursiveBFS(G,startingNode)
#Take unique set of heros, then add to graph one node at a time

#For each list of heros in relatedHeros, take the cartesian product of each array
#with itself to get a list of tuples. ['Jack','Kevin']*['Jack','Kevin']
#will get [('Jack','Kevin'),('Jack','Jack'),('Kevin','Jack'),('Kevin','Kevin')]
#Now how to remove self loops....Directed graph?.....