
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


# In[4]:

source_list = [u'CAPTAIN AMERICA', u'MISS THING/MARY', u'ORWELL']
touched_rec = []
for source_node in source_list:
    curr_nodes = do_bfs2(sc, source_node, hero_graph, n_parts)

    # Count the number of touched nodes at each iteration
    touched_rec.append(curr_nodes.map(lambda x: (x[1], x[0]))
                       .countByKey().items())


# In[5]:

touched_rec


# In[10]:

# Save number of touched nodes at each iteration in file
with open('P4.txt', 'w') as f:
    f.write('Number of touched nodes at each iteration: (distance, # of heroes)\n\n')
    for source_i, source_node in enumerate(source_list):
        f.write(source_node)
        f.write(':')
        f.write("{}".format(touched_rec[source_i]))
        f.write('\n\n')

