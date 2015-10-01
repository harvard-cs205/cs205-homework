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

#### Create the network ####
num_partitions = 40

# Assumes all links are **symmetric**
links_raw_data = sc.textFile('links-simple-sorted.txt', minPartitions=num_partitions)
titles_raw_data = sc.textFile('titles-sorted.txt', minPartitions=num_partitions)

import re

def get_links(x):
    split = re.findall(r"[\w']+", x)
    parent = int(split[0])
    children = [int(z) for z in split[1:]]
    parent_to_children = [(parent, z) for z in children]
    children_to_parent = [(z, parent) for z in children]
    return parent_to_children + children_to_parent

all_links = links_raw_data.flatMap(get_links)
node_then_all_links = all_links.groupByKey()
# Remove non-unique links
node_then_all_links_expanded = node_then_all_links.map(lambda x: (x[0], list(set(x[1]))))

network_rdd = node_then_all_links_expanded

### Run connected component code on network ###
from HW1.network_commands import Connected_Components

connector = Connected_Components(sc, network_rdd)
connector.run_until_converged()
# Print number of connected components at the end
print 'Number of unique groups:' , connector.get_num_unique_groups()

# Get the connected component with the biggest number of nodes
nodes_per_index = connector.get_number_of_nodes_per_index()
collected_nodes_per_index = nodes_per_index.collect()
num_nodes = map(lambda x: x[1], collected_nodes_per_index)

print 'Biggest group:' ,  np.sort(num_nodes)[-1]