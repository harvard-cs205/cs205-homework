'''
import findspark
findspark.init()
'''

import pyspark
sc = pyspark.SparkContext(appName="P5a")

sc.setLogLevel('ERROR')
import re
import csv
from P5_bfs import *


#load in the list of characters and appearances in comic books
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

links_graph = links.map(lambda x: x.replace(":","").split(" "))
links_graph = links_graph.map(lambda x: (int(x[0]), list(map(int, list(set(x[1:]))))))

#load pages
page_list = page_names.zipWithIndex()

#increment by 1 for 1 indexing
page_list = page_list.map(lambda (x,y): (x,y+1))
reverse_page_list = page_list.map(lambda (x,y): (y,x))

print "Created graph"

harvard_index = page_list.lookup("Harvard_University")[0]
bacon_index = page_list.lookup("Kevin_Bacon")[0]

print harvard_index, bacon_index
path = search(links_graph, bacon_index, harvard_index, sc)
path.reverse()
print path

formatted_path = format_path(reverse_page_list, path)
print formatted_path
#search(links_graph, bacon_index, harvard_index, sc)

'''
print search(adjacency_list, 'CAPTAIN AMERICA', 'EVERETT, BILL', sc)
'''
