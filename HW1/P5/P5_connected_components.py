
# coding: utf-8

# In[ ]:

import pyspark 

# This code is borrowed from Ray's github
def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), set(dests))

sc = pyspark.SparkContext()
sc.setLogLevel('WARN')

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

n_parts = 256
# process links into (node #, [neighbor node #, neighbor node #, ...]
neighbor_graph = links.map(link_string_to_KV)
neighbor_graph = neighbor_graph.partitionBy(n_parts).cache()


# In[ ]:

# Generate the single link and double link graphs
reverse_graph = (neighbor_graph.flatMap(lambda x: [(y, x[0]) for y in x[1]])
                 .groupByKey()
                 .mapValues(set))
reverse_join = neighbor_graph.join(reverse_graph).cache()

# print('Creating single link graph')
# graph_sing_link = reverse_join.mapValues(lambda x: x[0].union(x[1]))

print('Creating doubly link graph')
graph_double_link = reverse_join.mapValues(lambda x: x[0].intersection(x[1])).cache()
print('Finished creating doubly link graph')


# In[ ]:

# Import functions derived from Marvel graph computations
from P5 import *

# connected_hist_sing_link = find_connect_comp(sc, graph_sing_link, n_parts)
connected_hist_doub_link = find_connect_comp(sc, graph_double_link, n_parts)


# In[ ]:

# print(len(connected_hist_sing_link), max([x[1] for x in connected_hist_sing_link]))
print(len(connected_hist_doub_link), max([x[1] for x in connected_hist_doub_link]))

