###################################################
### Problem 4 - Graph processing in Spark [25%] ###
### P4.py                                       ###
### Patrick Day 								###
### CS 205 HW1                                  ###
### Oct 4th, 2015								###
###################################################

########################
### Import Functions ###
########################
import numpy as np
import csv
#import string

from P4_bfs import * 

########################
### Define Functions ###
########################

### Import Source ###
p4_path = "/Users/pday/Dropbox/Harvard/Fall15/CS205/HW1/source.csv"
lines = []
with open(p4_path, 'rb') as csvfile:
    source_csv = csv.reader(csvfile)
    for row in source_csv:
        lines.append(row)

# Call graph builder function
heros_graph = build_graph(lines)

# All heros
capt_america = "CAPTAIN AMERICA"
miss_thing = "MISS THING/MARY"
orwell = "ORWELL"

# Run Breath First Search
capt_graph, capt_depth = ss_bfs(heros_graph, capt_america)
mist_graph, miss_depth = ss_bfs(heros_graph, miss_thing)
orwell_graph, orwell_depth = ss_bfs(heros_graph, orwell)

# Print results
print(capt_america, "depth is:", capt_depth, "with # nodes:", capt_graph.count())
print(miss_thing, "depth is:", miss_depth, "with # nodes:", mist_graph.count())
print(orwell, "depth is:", orwell_depth, "with # nodes:", orwell_graph.count())