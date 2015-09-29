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
num_partitions_to_use=50

links_raw_data = sc.textFile('links-simple-sorted.txt', minPartitions=num_partitions_to_use)
titles_raw_data = sc.textFile('titles-sorted.txt', minPartitions=num_partitions_to_use)

def get_links(x):
    split = re.findall(r"[\w']+", x)
    name = int(split[0])
    linked_to = split[1:]
    linked_to = [int(z) for z in linked_to]
    return (name, linked_to)

all_links = links_raw_data.map(get_links)
network_rdd = all_links # Using a common phrase between programs for sanity

# Get the lookup table
titles_with_index_offset = titles_raw_data.zipWithIndex()
index_then_title = titles_with_index_offset.map(lambda x: (x[1] + 1, x[0])) # One indexed!
title_then_index = index_then_title.map(lambda x: (x[1], x[0]))

# Import my code to find paths
from HW1.network_commands import Path_Finder

# Lookup start & end node
start_node = title_then_index.lookup('Harvard_University')[0]
end_node = title_then_index.lookup('Kevin_Bacon')[0]

# Optimize rdd's for use in code, use same # of partitions across code
title_then_index = title_then_index.sortByKey(numPartitions=num_partitions_to_use).cache()
index_then_title = index_then_title.sortByKey(numPartitions=num_partitions_to_use).cache()
network_rdd = network_rdd.sortByKey(numPartitions=num_partitions_to_use).cache()
Path_Finder.num_partitions = num_partitions_to_use

# Create Path_Finder and find the path!
finder = Path_Finder(sc, network_rdd, start_node, end_node)
finder.run_until_converged()

# Look at the path and print a couple of paths

path = finder.distance_rdd
path_rdd = sc.parallelize(path, num_partitions_to_use)

def get_path_forwards():
    chosen_parent = end_node
    path_back = [end_node]

    go = True
    path_forwards = None
    while go:
        potential_parents = path_rdd.lookup(chosen_parent)[0][1]
        # Let's make the parent choice random to get variability
        chosen_parent = np.random.choice(potential_parents)
        path_back.append(chosen_parent)
        if chosen_parent == start_node:
            go = False
    path_forwards = list(reversed(path_back))
    return path_forwards

print
print
print [index_then_title.lookup(z)[0] for z in get_path_forwards()]
print [index_then_title.lookup(z)[0] for z in get_path_forwards()]
print [index_then_title.lookup(z)[0] for z in get_path_forwards()]
print
print