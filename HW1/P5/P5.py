from P5_bfs import * 
from P5_connected_components import *
import pyspark
from pyspark import SparkContext

#Setup
sc = SparkContext("local[8]")
sc.setLogLevel("ERROR")

#loading in the data
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

#zip page names with index to get a table for us to reference which page is which
zpage_names = page_names.zipWithIndex().mapValues(lambda x: x+1)
#need to turn link rows into two elements and convert from strings to integers
links = links.map(lambda x: x.split(': ')).map(lambda x: (x[0],x[1])).mapValues(lambda x: x.split(" ")).flatMapValues(lambda x: x).map(lambda x: (int(x[0]),int(x[1]))).partitionBy(32)

Harvard_ID = zpage_names.lookup("Harvard_University")[0]
print Harvard_ID
Bacon_ID = zpage_names.lookup("Kevin_Bacon")[0]

#determine the shortest path between Harv->Bacon and Bacon->Harv
print bfs(Harvard_ID,links,sc,Bacon_ID)
print bfs(Bacon_ID,links,sc,Harvard_ID)


print connected_components(Harvard_ID,links,sc)

