
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

# create an RDD for looking up page names from numbers
# remember that it's all 1-indexed
page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()


# find Kevin Bacon
Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
# This should be [(node_id, 'Kevin_Bacon')]
assert len(Kevin_Bacon) == 1
Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id

# find Harvard University
Harvard_University = page_names.filter(lambda (K, V):
                                       V == 'Harvard_University').collect()
# This should be [(node_id, 'Harvard_University')]
assert len(Harvard_University) == 1
Harvard_University = Harvard_University[0][0]  # extract node id
    

# Import functions derived from Marvel graph computations
from P5 import *

# Run BFS from Kevin_Bacon
source_node = Kevin_Bacon
dest_node = Harvard_University
hero_graph = neighbor_graph

short_path = find_short_path(source_node, dest_node, hero_graph, n_parts, sc)

# Convert ids back to page names
short_path = [page_names.lookup(x)[0] for x in short_path]

print(short_path)

# Save shortest path to file
save_str_list(short_path, 'P5.txt')

