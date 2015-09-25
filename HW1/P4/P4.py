from pyspark import SparkContext

import networkx as nx
import matplotlib.pyplot as plt
import itertools
import pdb

from P4_bfs import nonRecursiveBFS,resetDistance

# sc = SparkContext()
# textFile = sc.textFile('source.csv')
# #Convert to |Hero Name|,|Comic| then take out the Comic and place on left, Hero on right. Later we will groupBy() using comic as key
# formatedTf = textFile.map(
# 	lambda x:(x.replace('"','|').split('|')[3].strip(),x.replace('"','|').split('|')[1].strip()))
# #We group by Comic, and the value of each key(comic) will be an array of heros 
# #in that comic
# groupedTf = formatedTf.groupByKey().map(lambda t:(t[0],list(t[1])))
# #Using N choose K method from itertools, I take pairs of heros from the same
# #comic, then flatten the tuples array. This data will be used as edges later on
# #I sort beforehand so we can eliminate scenarios such as (A,B),(B,A). We only want one of them
# heroRelationTuplesNoDup = groupedTf.flatMap(lambda t: list(itertools.combinations(sorted(t[1]),2)))
# #Remove duplicates by putting list in set
# heroTuplesNoDup = set(heroRelationTuplesNoDup.collect())
# #Take just the hero names from text file

# uniqueHeros = textFile.map(lambda x:x.replace('"','|').split('|')[1].strip())

# #Take the unique names
# uniqueHeroNames = set(uniqueHeros.collect())


# #Create graph
# G = nx.Graph()
# #add distance attr to each node
# for x in uniqueHeroNames:
# 	G.add_node(x,distance=-1)
# #Add edges
# for edge in heroTuplesNoDup:
# 	G.add_edge(edge[0],edge[1])

# #Draws graph
# # plt.figure(figsize=(20,20))
# # nx.draw_random(G,with_labels=False,node_size=1,width=0.1)
# # plt.savefig('graph.png')



# def shortestPath(Graph,startingNode,count):
# 	pdb.set_trace()
# 	children = []
# 	distances=[Graph.node[startingNode]['distance']]
# 	# if len(Graph.successors(startingNode)) == 0:
# 	# 	return Graph.node[startingNode]['distance']

# 	for c in Graph.neighbors(startingNode):
# 		if Graph.node[c]['distance'] == -1:
# 			Graph.node[c]['distance'] = count+1
# 			distances.append(shortestPath(Graph,c,count+1))
	
# 	return max(distances)
# 	#if no children, then return current node's distance
# #Reset distance of each node to 0



# startingNodes = ['CAPTAIN AMERICA']
# for n in startingNodes:
# 	resetDistance(G)
# 	print nonRecursiveBFS(G,n)

sc = SparkContext()
textFile = sc.textFile('source.csv',20)
#Convert to |Hero Name|,|Comic| then take out the Comic and place on left, Hero on right. Later we will groupBy() using comic as key
formatedTf = textFile.map(
	lambda x:(x.replace('"','|').split('|')[3].strip(),x.replace('"','|').split('|')[1].strip()))
#We group by Comic, and the value of each key(comic) will be an array of heros 
#in that comic
groupedTf = formatedTf.groupByKey().map(lambda t:(t[0],list(t[1])))
hero_HeroList = groupedTf.flatMap(lambda f:[(y,[x for x in f[1] if x != y]) for y in f[1]])
groupedHero_HeroList = hero_HeroList.groupByKey().map(lambda f:(f[0],list(set([inner for outer in list(f[1]) for inner in outer]))))
smallHeroGroup = sc.parallelize(groupedHero_HeroList.filter(lambda a:a[0]=='CAPTAIN AMERICA').collect())
#print len(groupedHero_HeroList.lookup('CAPTAIN AMERICA')[0])

a=[]
# for i in range(2):
# 	a = smallHeroGroup.flatMap(
# 		lambda x:[(y,[a for a in x[1] if a != y]) for y in x[1]]).groupByKey().map(
# 		lambda f:(f[0],list(set([inner for outer in f[1] for inner in outer]))))
# 	smallHeroGroup = a.join(groupedHero_HeroList).map(
# 		lambda x:(x[0],list(x[1]))).map(
# 		lambda f:(f[0],list(set([inner for outer in f[1] for inner in outer]))))

# 	print smallHeroGroup.count()

for i in range(3):
	a = smallHeroGroup.flatMap(
		lambda x:[(y,[]) for y in x[1]]).reduceByKey(lambda x,y:[])
	smallHeroGroup = a.join(groupedHero_HeroList).map(
		lambda x:(x[0],list(x[1]))).map(
		lambda f:(f[0],list(set([inner for outer in f[1] for inner in outer]))))

	print smallHeroGroup.count()






# print b.count()
# c = b.flatMap(lambda x:[(y,[a for a in x if a != y]) for y in x])
# d = c.join(groupedHero_HeroList).map(lambda x:(x[0],list(x[1]))).map(lambda f:(f[0],list(set([inner for outer in f[1] for inner in outer]))))
# print d.count()
#print a.join(groupedHero_HeroList).take(1)
#print smallHeroGroup.reduce(lambda x,y:len(x[1])+len(y[1]))
#print smallHeroGroup.take(2)











