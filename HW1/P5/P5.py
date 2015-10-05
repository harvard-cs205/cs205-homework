from P5_BFS import *
from  pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [to for to in dests.split(' ')]
    return (src, dests)

page_names = page_names.zipWithIndex().mapValues(lambda V: V+1)
links = links.map(link_string_to_KV)
edgelist = links.flatMap(lambda (K,V): [(K,V[i]) for i in range(len(V))])
edgelist.partitionBy(256).cache()
edgelist.take(1) 


node1 = str(page_names.lookup(u'Kevin_Bacon')[0])
node2 = str(page_names.lookup("Harvard_University")[0])

print "Edgelist ready"

print bfs_look(edgelist,node1,node2,10,sc)
print bfs_look(edgelist,node2,node1,10,sc)

#