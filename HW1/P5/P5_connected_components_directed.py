import seaborn as sns
sns.set_context('poster', font_scale=1.25)
import findspark as fs
fs.init()
import pyspark as ps
import multiprocessing as mp
import numpy as np
import re

# Assumes local options are already set in conf file...or else this explodes
config = ps.SparkConf()
config = config.setAppName('wiki_solver')
sc = ps.SparkContext(conf=config)

# Create the network
num_partitions = 40

### Obtaining only *symmetric* links from the underlying data ###

links_raw_data = sc.textFile('links-simple-sorted.txt')
titles_raw_data = sc.textFile('titles-sorted.txt')

import re

def get_links(x):
    split = re.findall(r"[\w']+", x)
    parent = int(split[0])
    children = [int(z) for z in split[1:]]

    return (parent, children)

parent_child_links = links_raw_data.map(get_links).partitionBy(num_partitions)

def get_reversed_links(x):
    parent = x[0]
    children = x[1]
    return ((z, parent) for z in children)

all_reversed_links = parent_child_links.flatMap(get_reversed_links)

child_then_parents = all_reversed_links.groupByKey(num_partitions)
child_then_parents_expanded = child_then_parents.map(lambda x: (x[0], list(x[1])), preservesPartitioning=True)

joined_links = parent_child_links.join(child_then_parents_expanded)

def get_acceptable_links(x):
    parent = x[0]
    list1 = x[1][0]
    list2 = x[1][1]
    symmetric_links = set(list1).intersection(list2)
    symmetric_links = list(symmetric_links)
    # Many nodes are unlinked to anything...so they get their own index!
    # They *are* a connected component.
    return (parent, symmetric_links)

symmetric_links = joined_links.map(get_acceptable_links, preservesPartitioning=True)

network_rdd = symmetric_links

# Analyze the links using my connected components class

from HW1.network_commands import Connected_Components

connector = Connected_Components(sc, network_rdd)
connector.run_until_converged()

# Get the biggest group
num_rdd = connector.get_number_of_nodes_per_index()
id_and_num = num_rdd.collect()
num_nodes = map(lambda x: x[1], id_and_num)
print 'Biggest group:' ,  np.sort(num_nodes)[-1]