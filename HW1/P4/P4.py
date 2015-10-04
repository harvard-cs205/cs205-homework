from pyspark import SparkContext

import networkx as nx
import matplotlib.pyplot as plt
import itertools
import pdb

from P4_bfs import nonRecursiveBFS,resetDistance

sc = SparkContext()
textFile = sc.textFile('source.csv',20)
# Convert to |Hero Name|,|Comic| then take out the Comic and place on left, Hero on right. Later we will groupBy() using comic as key
# (Comic,Hero)
# formatedTf = textFile.map(
# 	lambda x:(x.replace('"','|').split('|')[3].strip(),x.replace('"','|').split('|')[1].strip()))
# #(Comic,[Heros])
# groupedTf = formatedTf.groupByKey().map(lambda t:(t[0],list(t[1])))
# #[H,[Heros in same comic as H]]
# hero_HeroList = groupedTf.flatMap(lambda f:[(y,[x for x in f[1] if x != y]) for y in f[1]])
# #(H,[all heros related to H]) 
# groupedHero_HeroList = hero_HeroList.groupByKey().map(lambda f:(f[0],list(set([inner for outer in list(f[1]) for inner in outer]))))

# smallHeroGroup = sc.parallelize(groupedHero_HeroList.filter(lambda a:a[0]=='CAPTAIN AMERICA').collect())
# # smallHeroGroup = sc.parallelize(groupedHero_HeroList.filter(lambda a:a[0]=='ORWELL').collect())
# #smallHeroGroup = sc.parallelize(groupedHero_HeroList.filter(lambda a:a[0]=='MISS THING/MARY').collect())
# #print len(groupedHero_HeroList.lookup('CAPTAIN AMERICA')[0])

# a=[]
# prevCount = 0
# iterations = 0
# for i in range(10):
# 	a = smallHeroGroup.flatMap(
# 		lambda x:[(y,[]) for y in x[1]]).reduceByKey(lambda x,y:[])
# 	smallHeroGroup = a.join(groupedHero_HeroList).map(
# 		lambda x:(x[0],list(x[1]))).map(
# 		lambda f:(f[0],list(set([inner for outer in f[1] for inner in outer]))))
# 	newCount = smallHeroGroup.count()
# 	if newCount==prevCount:
# 		break;
# 	else:
# 		iterations=iterations+1
# 		prevCount = newCount
# print iterations
# print prevCount
#################################
#.map(lambda x:(x.replace('"','|').split('|')[3].strip(),x.replace('"','|').split('|')[1].strip()))
startNode = 'CAPTAIN AMERICA'		
#startNode = 'MISS THING/MARY'
graph = (
	textFile
	.map(lambda x:(x.split('"')[3],x.split('"')[1]))
	.groupByKey()
	.map(lambda t:(t[0],list(t[1])))
	.flatMap(lambda f:[(y,tuple([x for x in f[1] if x != y])) for y in f[1]])
	.groupByKey().map(lambda f:(f[0],10000,frozenset([inner for outer in f[1] for inner in outer]),False))
	.map(lambda x:(x[0],0,x[2],True) if x[0] == startNode else x)
	.cache()
)

accum = sc.accumulator(0)
accum2 = sc.accumulator(0)

def extraArgs(neighbors,distance):
	def updateNode(n):
		if n[0] in neighbors and n[1] > distance+1:
			# if n[3] == False:
			# 	accum2.add(1)
			if n[3] == False:
				accum.add(1)
				return (n[0],distance+1,n[2],True)
			else:
				return (n[0],distance+1,n[2],n[3])
		else:
			return n
	return updateNode

def f(neighbors):
	return lambda y:y[3]==False and y[0] in neighbors
	
greyNodes= set(graph.filter(lambda x:x[0]==startNode).collect())
# print next(iter(greyNodes))[1]
for i in range(4):
	neighbors = set([])
	print 'greynodes count=',len(greyNodes)
	for x in greyNodes.copy():
		neighbors.update(x[2])
		greyNodes.remove(x)
	greyNodes.update(graph.filter(f(neighbors)).collect())
	graph = graph.map(extraArgs(neighbors,i))
	graph.count()


# 		# print 'neighbors=',neighbors
# 		# print len(graph.filter(lambda y:y[0] in neighbors and y[3]==False).collect())
# print len(graph.filter(lambda x:x[3]==True).collect())

#print 'accum',accum.value
# print 'accum2',accum2.value			
# for x in graph.filter(lambda x:x[1]<100).collect():
# 	if int(x[1])> 1:
# 		print x[0].encode('utf-8'),x[1]
#print graph.filter(lambda x:x[1]<100).collect()
# endNode = '5'
# print graph.filter(lambda y:y[0]==endNode).collect()


