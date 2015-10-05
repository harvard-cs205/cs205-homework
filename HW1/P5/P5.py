from P5_bfs import * 
from P5_connected_components import *
import pyspark
from pyspark import SparkContext

#Setup
sc = SparkContext()
sc.setLogLevel("ERROR")

#loading in the data
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
#links = sc.textFile('test.txt')
#page_names = sc.textFile('test2.txt')

#zip page names with index to get a table for us to reference which page is which
page_names = page_names.zipWithIndex().map(lambda (K, V): (V+1, K)).cache()
#need to turn link rows into two elements and convert from strings to integers
links = links.map(lambda x: x.split(': ')).map(lambda x: (x[0],x[1])).mapValues(lambda x: x.split(" ")).flatMapValues(lambda x: x).map(lambda x: (int(x[0]),int(x[1]))).partitionBy(128).cache()

#find harvard and kevin bacon
Harvard_ID = page_names.filter(lambda (K,V): V == 'Harvard_University').collect()[0][0]
Bacon_ID = page_names.filter(lambda (K,V): V == 'Kevin_Bacon').collect()[0][0]
print Harvard_ID
print Bacon_ID

#determine shortest path between Harv->Bacon and Bacon->Harv

print bfs(Bacon_ID,links,sc,Harvard_ID)
print bfs(Harvard_ID,links,sc,Bacon_ID)
#print connected_components(Harvard_ID,links,sc)


