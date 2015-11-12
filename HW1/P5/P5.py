#######################################################################
### Problem 5 - Larger-scale Graph Processing with AWS & Spark[30%] ###
### P4_bfc.py														###
### Patrick Day 													###
### CS 205 HW1                                  					###
### Oct 4th, 2015													###
#######################################################################

########################
### Import Functions ###
########################
import pyspark
import numpy as np
import csv
import string
from operator import add

import pickle
import re
import itertools

import os
sc = pyspark.SparkContext()
sc.setLogLevel('WARN')

########################
### Define Functions ###
########################
def copartitioned(RDD1, RDD2):
	"check if two RDDs are copartitioned"
	return RDD1.partitioner == RDD2.partitioner

def ss_bfs(page_ind, links_rdd, start_node, end_node):
	start_page = [(start_node, 0)]
	start_rdd = sc.parallelize(start_page).partitionBy(256)
	end_page = end_node
	
	depth = 0
	stop_critieria = 4
	print("BFS for start_node")
	while depth < stop_critieria:
	    print("Itneration:", depth)
	    # Filter Parents and join with link_rdd and check copartition
	    parent_rdd = start_rdd.filter(lambda x: x[1] == depth).join(links_rdd).mapValues(lambda x: x[1])
	    assert copartitioned(start_rdd, parent_rdd)
	    
	    # Flatmap childred and ensure doesn't exist in current start_rdd
	    child_rdd = parent_rdd.flatMap(lambda x: x[1]).map(lambda x: (x, 1)).subtractByKey(start_rdd).distinct().partitionBy(256)
	    assert copartitioned(start_rdd, child_rdd) 

	    # Add depth to the child
	    next_child_rdd = child_rdd.map(lambda x: (x[0], depth + 1), True)
	    assert copartitioned(start_rdd, next_child_rdd) 

	    # Add to the final RDD
	    start_rdd = start_rdd.union(next_child_rdd).cache()
	 
	    depth += 1
	    print()

	return start_rdd 

def find_shortest(start_rdd, page_ind, links_rdd, start_node, end_node, depth):
	# Explore Parents of children (go back up the tree)
	canidate_parent = start_rdd.filter(lambda x: x[1] == depth-1).partitionBy(256)
	
	# Join with all other links and get values
	canidate_parent_of_child = canidate_parent.join(links_rdd).mapValues(lambda x: x[1])
	
	# Find Harvard's connection in the children, get the key
	found_parent = canidate_parent_of_child.filter(lambda x: end_node in x[1]).keys()

	# Collect Next Node Step
	next_parent = found_parent.collect()
	connect_node = page_ind.filter(lambda (x,y): y == int(next_parent[0])).collect()

	# Put output together in list
	connection_list = [start_node, end_node, 'via:', connect_node[0][0]]

	return connect_node

##################################
### Import data and Preprocess ###
##################################

# Import Sources
print("Program Started")

wiki_links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)#, use_unicode=False)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)#, use_unicode=False)

# Add Index
page_ind = page_names.zipWithIndex().map(lambda (n, id): (n, id+1))

# Find Kevin Bacon! and Harvard...
kevin_bacon = page_ind.filter(lambda (x,y): x == "Kevin_Bacon").collect()
kevin_bacon_name = unicode(kevin_bacon[0][1])
harvard_uni = page_ind.filter(lambda (x,y): x == "Harvard_University").collect()
harvard_uni_name = unicode(harvard_uni[0][1])

# Preprocess name with regular expressions
re_format = re.compile(r'[%s\s]+' % re.escape(string.punctuation))
links_split = wiki_links.map(lambda x: re_format.split(x))
links_rdd = links_split.map(lambda x: (x[0], x[1:])).partitionBy(256).cache()

########################################
### Kevin's Shortest Path to Harvard ###
########################################

# Run bfs function for kevin bacon
kevin_rdd = ss_bfs(page_ind, links_rdd, kevin_bacon_name, harvard_uni_name)

# Print out Kevin's Information #
kevin_list = kevin_rdd.collect()
pickle.dump(kevin_list, open("kevin_node_graph.pkl", "wb"))

#file = open("kevin_node_graph.txt", "w")
#for line in kevin_list:
#  	file.write("%s\n" % item)
#file.close()

### Find shortest path ###
# Explore Parents of children (go back up the tree)
kevin_path_to_harvard = find_shortest(kevin_rdd, page_ind, links_rdd, kevin_bacon_name, harvard_uni_name, 2)

all_poss_paths = [kevin_bacon_name, harvard_uni_name, 'via:', kevin_path_to_harvard[0][0]]
print("Kevin's shortest Path:", all_poss_paths)

pickle.dump(connect_node, open("kevin_to_harvard_shortest_path.pkl", "wb"))
#file = open("kevin_shortest_path_harvard_.txt", "w")
#for line in kevin_path_to_harvard:
#	file.write("%s\n" % line)
#file.close()

###################################################
### Find Harvard's shortest path to Kevin Bacon ###
###################################################
harvard_rdd = ss_bfs(page_ind, links_rdd, kevin_bacon_name, harvard_uni_name)

### Print out Harvard's Information ###
harvard_list = harvard_rdd.collect()
pickle.dump(harvard_list, open("harvard_node_graph.pkl", "wb"))

#file = open("harvard_node_graph.txt", "w")
#for line in harvard_list:
#  	file.write("%s\n" % line)
#file.close()

### Find shortest path ###
harvard_path_to_kevin = find_shortest(harvard_rdd, page_ind, links_rdd, harvard_uni_name, kevin_bacon_name, 3)

# Keep going down the path, encode the next_parent
child_of_cand_parent = unicode(next_parent[0][1])
harvard_path_to_kevin_via_step = find_shortest(harvard_rdd, page_ind, links_rdd, harvard_path_to_kevin, kevin_bacon_name, 2)

# Put all steps in a list and pickle dump
short_path_harvard = [harvard_path_to_kevin + harvard_path_to_kevin_via_step]
pickle.dump(short_path_harvard, open("harvard_shortest_path.pkl", "wb"))

#file = open("harvard_shortest_path.txt", "w")
#for line in kevin_list:
#  	file.write("%s\n" % item)
#file.close()
