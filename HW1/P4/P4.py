from P4_bfs import *
import findspark
findspark.init('/home/toby/spark')

import pyspark

sc = pyspark.SparkContext(appName="Spark 2")
myfile = "source.csv"

graph = sc.textFile(myfile)

def parser(string):
    words = string.split("\"")
    return words[3].strip(), words[1].strip()

graph = graph.map(parser)
graph = graph.join(graph)
graph = graph.filter(lambda element: element[1][0]!=element[1][1])

graph = graph.map(lambda element: (element[1][0], set([element[1][1]])))
graph = graph.reduceByKey(lambda a, b: a|b)
graph = graph.map(lambda element: (element[0], list(element[1])))
graph = graph#.partitionBy(100)
path = graph.collectAsMap()


characters = ["CAPTAIN AMERICA", "MISS THING/MARY", "ORWELL"]
num_nodes = []
for character in characters:
    num_nodes.append(BFS(character, graph))
for character in characters:
    print character, "has"
print num_nodes

#print graph.map(lambda x: x).lookup("M")
#print graph.filter(lambda KV: KV[0]=="MISS THING/MARY").collect()
#print "Number of nodes visited is", num_nodes 
#print graph.map(lambda x: x).lookup("MISS THING/MARY")
#print path["MISS THING/MARY"]
#for Key, Value in a.iteritems():
#    print Key, ":", Value
#    print
#    print
#print path['STERLING'] 
#print path['PANTHER CUB/']
#print path['SWORDSMAN IV/']
#print path['AMAZO-MAXI-WOMAN/']
#print path['DARLEGUNG, GEN.']
