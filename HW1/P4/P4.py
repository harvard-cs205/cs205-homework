from pyspark import SparkContext

import networkx as nx
import matplotlib.pyplot as plt
import itertools
import pdb

from P4_bfs import nonRecursiveBFS,resetDistance

sc = SparkContext()
textFile = sc.textFile('source.csv',20)
#Convert to |Hero Name|,|Comic| then take out the Comic and place on left, Hero on right. Later we will groupBy() using comic as key
#(Comic,Hero)
formatedTf = textFile.map(
	lambda x:(x.replace('"','|').split('|')[3].strip(),x.replace('"','|').split('|')[1].strip()))
#(Comic,[Heros])
groupedTf = formatedTf.groupByKey().map(lambda t:(t[0],list(t[1])))
#[H,[Heros in same comic as H]]
hero_HeroList = groupedTf.flatMap(lambda f:[(y,[x for x in f[1] if x != y]) for y in f[1]])
#(H,[all heros related to H]) 
groupedHero_HeroList = hero_HeroList.groupByKey().map(lambda f:(f[0],list(set([inner for outer in list(f[1]) for inner in outer]))))

smallHeroGroup = sc.parallelize(groupedHero_HeroList.filter(lambda a:a[0]=='CAPTAIN AMERICA').collect())
# smallHeroGroup = sc.parallelize(groupedHero_HeroList.filter(lambda a:a[0]=='ORWELL').collect())
#smallHeroGroup = sc.parallelize(groupedHero_HeroList.filter(lambda a:a[0]=='MISS THING/MARY').collect())
#print len(groupedHero_HeroList.lookup('CAPTAIN AMERICA')[0])

a=[]
prevCount = 0
iterations = 0
for i in range(10):
	a = smallHeroGroup.flatMap(
		lambda x:[(y,[]) for y in x[1]]).reduceByKey(lambda x,y:[])
	smallHeroGroup = a.join(groupedHero_HeroList).map(
		lambda x:(x[0],list(x[1]))).map(
		lambda f:(f[0],list(set([inner for outer in f[1] for inner in outer]))))
	newCount = smallHeroGroup.count()
	if newCount==prevCount:
		break;
	else:
		iterations=iterations+1
		prevCount = newCount
print iterations

