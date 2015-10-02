
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

# Load marvel comic data
dat = sc.textFile('../../DataSources/source.csv')

# Remove quotations and split issue & hero name
dat_split = dat.map(lambda x: x[1:-1].split('","'))

# Mapping between issue->hero
dat_comic = dat_split.map(lambda x: (x[1], x[0]))

# Mapping between issue->hero group
comic_key = (dat_split.map(lambda x: (x[1], {x[0]}))
             .reduceByKey(lambda a, b: a.union(b)))

# Mapping between hero-> hero group
comic_hero_join = dat_comic.join(comic_key).map(lambda x: x[1])

# Aggregate acquaintances of a single hero across multiple
hero_graph = (comic_hero_join.reduceByKey(lambda a, b: a.union(b))).cache()
n_parts = 20
hero_graph = hero_graph.partitionBy(n_parts, hash)


# In[4]:

source_list = [u'CAPTAIN AMERICA', u'MISS THING/MARY', u'ORWELL']
touched_rec = []
for source_node in source_list:
    curr_nodes = do_bfs2(sc, source_node, hero_graph)

    # Count the number of touched nodes at each iteration
    touched_rec.append(curr_nodes.map(lambda x: (x[1], x[0]))
                       .countByKey().items())


# In[6]:

curr_nodes


# In[6]:

touched_rec[0]


# In[ ]:

with open("P3.txt", "w") as text_file:
    text_file.write("{}".format(max_anagram))


# In[10]:

# Save number of touched nodes at each iteration in file
with open('P4.txt', 'w') as f:
    f.write('Number of touched nodes at each iteration: (distance, # of heroes)\n\n')
    for source_i, source_node in enumerate(source_list):
        f.write(source_node)
        f.write(':')
        f.write("{}".format(touched_rec[source_i]))
        f.write('\n\n')


# In[ ]:



