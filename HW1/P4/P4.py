# import findspark
# findspark.init()
import pyspark
import numpy as np

from P4_bfs import *


# We lode the data and create the graph

superheroes = sc.textFile('./P4/source.csv')

ByArticle = superheroes.map(lambda line: (line.split('","')[1][:-1], line.split('","')[0][1:]))

joined = ByArticle.join(ByArticle).values().groupByKey()

def remove_member(setHeroes, member):
    setHeroes.remove(member)
    return list(setHeroes)

Hero_graph = joined.map(lambda x: (x[0], remove_member(set(x[1]),x[0])))

N = 8

#We calculate the number of nodes touched for the given superheroes
[nodes_touched_CA, _] = BFS(Hero_graph, 'CAPTAIN AMERICA', N)
[nodes_touched_O, _] = BFS(Hero_graph, 'ORWELL', N)
[nodes_touched_MT, _] = BFS(Hero_graph, 'MISS THING/MARY', N)

#The number of nodes touched for:
#- Captain America is 6408
#- Orwell is 9
#- Miss thing/Mary is 7


