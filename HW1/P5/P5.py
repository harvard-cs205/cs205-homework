from P5_BFS import *
from  pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [to.encode('utf-8') for to in dests.split(' ')]
    return (src.encode('utf-8'), dests)

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)
page_names = page_names.zipWithIndex().mapValues(lambda V: V+1)
links = links.map(lambda x: link_string_to_KV(x))
edgelist = links.flatMap(lambda (K,V): [(K,V[i]) for i in range(len(V))])
edgelist.take(1) 
edgelist.partitionBy(128).cache()


node_Kevin = page_names.lookup(u'Kevin_Bacon')
node_harvard = page_names.lookup("Harvard_University")

print "Edgelist ready"
print node_Kevin

print bfs_look2(edgelist,node_Kevin,node_harvard,10,sc)