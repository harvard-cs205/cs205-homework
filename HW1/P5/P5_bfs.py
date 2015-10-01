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
num_partitions_to_use=40

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

# Optimize rdd's for lookups, according to documentation
title_then_index = title_then_index.sortByKey(numPartitions=num_partitions_to_use).cache()
index_then_title = index_then_title.sortByKey(numPartitions=num_partitions_to_use).cache()

# Create Path_Finder and find the path!
finder = Path_Finder(sc, network_rdd, start_node, end_node) #network_rdd is cached inside
finder.run_until_converged()

# Look at the path and print a couple of paths

def get_path_forwards():
    return [index_then_title.lookup(z)[0] for z in finder.get_random_path()]


print
print
print get_path_forwards()
print get_path_forwards()
print get_path_forwards()
print
print
