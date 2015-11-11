
# coding: utf-8

# In[1]:

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark 2")

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm


# In[2]:

from P4_bfs import *


# In[3]:

# Number of partitions we want
n_parts = 20

# Import data and make hero graph
hero_graph = make_hero_graph('../../DataSources/source.csv', sc, n_parts)


# In[7]:

source_list = [u'CAPTAIN AMERICA', u'MISS THING/MARY', u'ORWELL']
touched_rec = []
all_node_hist = []
for source_node in source_list:
    node_hist = do_bfs2(sc, source_node, hero_graph, n_parts, None)
    
    all_node_hist.append(node_hist)
    # Count the number of touched nodes at each iteration
    touched_rec.append(node_hist.map(lambda x: (x[1], x[0]))
                       .countByKey().items())


# In[23]:

all_node_hist[0].filter(lambda x: x[1] == 2).take(3)


# In[27]:

(do_bfs2(sc, source_node, hero_graph, n_parts, u'TRITON')
 .filter(lambda x: x[1] == 2).take(3))


# In[26]:

(do_bfs2(sc, source_node, hero_graph, n_parts, u'PENTIGAAR')
 .filter(lambda x: x[1] == 3).take(3))


# In[10]:

# Save number of touched nodes at each iteration in file
with open('P4.txt', 'w') as f:
    f.write('Number of touched nodes at each iteration: (distance, # of heroes)\n\n')
    for source_i, source_node in enumerate(source_list):
        f.write(source_node)
        f.write(':')
        f.write("{}".format(touched_rec[source_i]))
        f.write('\n\n')


# In[28]:

def find_short_path(source_node, dest_node, hero_graph, n_parts):
    node_hist = do_bfs2(sc, source_node, hero_graph, n_parts, dest_node)
    # Find what the shortest distance from source to destination is
    dest_dist = node_hist.lookup(dest_node)[0]
    dest_set = {dest_node}

    shortest_path_options = [[dest_node]]
    for dist_i in range(dest_dist - 1, 0, -1):
        # Working backward from the destination, 
        # find all the nodes that are pointing to it
        connected_nodes = (hero_graph.filter(
            lambda x: len(dest_set.intersection(x[1])) > 0))

        # Find all the nodes along the shortest paths 
        # according to our BFS search 
        shortest_path_nodes = node_hist.filter(lambda x: x[1] == dist_i)

        # The intersection of these are the nodes that are potentially
        # along the shortest path between the source and destination nodes.
        possible_path_node = connected_nodes.join(shortest_path_nodes)
        dest_set = set(possible_path_node.keys().collect())

        shortest_path_options.append(list(dest_set))
    shortest_path_options.append([source_node])
    # Return one possible shortest path
    return [options[0] for options in reversed(shortest_path_options)]


# In[29]:

source_node = u'CAPTAIN AMERICA'
# dest_node = u'KLEIN, SUMMER'
dest_node = u'PENTIGAAR'


# In[17]:

all_node_hist[0].filter(lambda x: x[1] == 2).take(3)


# In[30]:

short_path = find_short_path(source_node, 
                             dest_node, 
                             hero_graph, 
                             n_parts)


# In[31]:

short_path


# In[119]:

short_path = find_short_path(source_node, 
                             dest_node, 
                             node_hist, 
                             hero_graph)


# In[120]:

short_path


# In[18]:

hero_graph.take(2)


# In[29]:

reverse_graph = (hero_graph.flatMap(lambda x: [(y, x[0]) for y in x[1]])
                 .groupByKey()
                 .mapValues(set))


# In[32]:

reverse_join = hero_graph.join(reverse_graph)


# In[34]:

graph_sing_link = reverse_join.mapValues(lambda x: x[0].union(x[1]))


# In[38]:

graph_double_link = reverse_join.mapValues(lambda x: x[0].intersection(x[1]))


# In[36]:

graph_sing_link.take(3)


# In[39]:

graph_double_link.take(3)


# In[16]:

hero_graph.take(2)


# In[5]:

# Function that finds the number of connected components in a graph
def find_connect_comp(sc, hero_graph, n_parts):
    import time
    # Get list of all heroes
    hero_list = hero_graph.keys()
    curr_hero_graph = hero_graph
    connected_count = 0 # Count the number of connected components
    connect_hist = [] # Record information about connected components

    while hero_list.count() > 0:
        t1 = time.time()
        print(connected_count)

        # Take a node that hasn't been explored yet
        source_node = hero_list.first()

        # Do BFS and recover all touched nodes
        print('Running BFS...')
        search_history = do_bfs2(sc, source_node, curr_hero_graph, n_parts)
        search_history = search_history.partitionBy(n_parts, hash).cache()

        # Determine remaining untouched nodes and repeat
        print('Pruning nodes...')
        hero_list = hero_list.subtract(search_history.keys(), n_parts)

        # hero_list = hero_list.partitionBy(n_parts, hash).cache()

        # Remove nodes that have been explored 
        curr_hero_graph = curr_hero_graph.subtractByKey(search_history)
        curr_hero_graph = curr_hero_graph.partitionBy(n_parts, hash).cache()

        t2 = time.time()
        print((search_history.count(), curr_hero_graph.count(), t2 - t1))
        connect_hist.append((connected_count, search_history.count(), t2 - t1))
        connected_count = connected_count + 1
    return connect_hist


# In[6]:

connected_hist = find_connect_comp(sc, hero_graph, n_parts)


# In[8]:

connected_hist


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

# process links into (node #, [neighbor node #, neighbor node #, ...]
neighbor_graph = links.map(link_string_to_KV)

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
n_parts = 20
source_node = Kevin_Bacon
dest_node = Harvard_University
hero_graph = neighbor_graph

node_hist = do_bfs_aws(sc, source_node, neighbor_graph, n_parts)


# In[ ]:

source_node = Kevin_Bacon
dest_node = Harvard_University
hero_graph = neighbor_graph


# In[13]:

{source_node}


# In[15]:

set([source_node])

