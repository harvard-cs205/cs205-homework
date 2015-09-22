from pyspark import SparkContext

import networkx as nx
import matplotlib.pyplot as plt
import itertools

sc = SparkContext()
textFile = sc.textFile('source.csv')
formatedTf = textFile.map(
	lambda x:(x.replace('"','|').split('|')[3],x.replace('"','|').split('|')[1]))
groupedTf = formatedTf.groupByKey().map(lambda t:(t[0],list(t[1])))
#Has too many duplicates, n choose k is better
#heroRelationTuplesNoDuplicates=groupedTf.map(lambda t:[x for x in np.transpose([np.tile(t[1], len(t[1])), np.repeat(t[1], len(t[1]))]) if x[0]!=x[1]])
heroRelationTuplesNoDup = groupedTf.flatMap(lambda t: list(itertools.combinations(sorted(t[1]),2)))
heroTuplesNoDup = set(heroRelationTuplesNoDup.collect())
uniqueHeros = textFile.map(
	lambda x:x.replace('"','|').split('|')[1])
uniqueHeroNames = set(uniqueHeros.collect())



print len(heroTuplesNoDup)
#Draws nodes
G = nx.Graph()
for x in uniqueHeroNames:
	G.add_node(x)
for edge in heroTuplesNoDup:
	G.add_edge(edge[0],edge[1])
plt.figure(figsize=(20,20))
# pos=nx.random_layout(G)
nx.draw_random(G,with_labels=False,node_size=1,width=0.1)

plt.savefig('graph.png')



#Take unique set of heros, then add to graph one node at a time

#For each list of heros in relatedHeros, take the cartesian product of each array
#with itself to get a list of tuples. ['Jack','Kevin']*['Jack','Kevin']
#will get [('Jack','Kevin'),('Jack','Jack'),('Kevin','Jack'),('Kevin','Kevin')]
#Now how to remove self loops....Directed graph?.....