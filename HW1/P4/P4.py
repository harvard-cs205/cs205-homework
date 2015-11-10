

import pickle
import string
import re
import urllib2
import csv
import itertools
import math as ma
import os
import P4_bfs


if __name__ == '__main__':


	partition = 16

	output_rdd_captain_america = P4_bfs.bfs(start_node = 'CAPTAIN AMERICA', number_interations = 10, partition = partition)

	node_distances_captain_america = output_rdd_captain_america.bfs_search()

	print 'The max node distance for Captain America is', node_distances_captain_america[-1][1]

	print 'The number of nodes in Captain Americas graph is', len(node_distances_captain_america)

	pickle.dump(node_distances_captain_america, open( "P4_captain_america.pkl", "wb" ) )


	output_rdd_mary = P4_bfs.bfs(start_node = 'MISS THING/MARY', number_interations = 10, partition = partition)

	node_distances_mary = output_rdd_mary.bfs_search()

	print 'The max node distance for Mary is', node_distances_mary[-1][1]

	print 'The number of nodes in Mary graph is', len(node_distances_mary)

	pickle.dump(node_distances_mary, open( "P4_mary.pkl", "wb" ) )


	output_rdd_orwell = P4_bfs.bfs(start_node = 'ORWELL', number_interations = 10, partition = partition)

	node_distances_orwell = output_rdd_orwell.bfs_search()

	print 'The max node distance for Orwell is', node_distances_orwell[-1][1]

	print 'The number of nodes in Orwell graph is', len(node_distances_orwell)

	pickle.dump(node_distances_orwell, open( "P4_orwell.pkl", "wb" ) )





