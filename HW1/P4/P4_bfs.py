import pdb


def mybfs(nodes, name):

	# Create a rdd with all characters as keys and 1 as values
	# allchar has the keys for the characters not touched by the search
	allchar = nodes.map(lambda x: (x[0], 1)  )

	# Total number of characters
	num_char_ini = len(allchar.collect())

	# First find node 0 to start the algorithm
	node0 = nodes.filter(lambda x: x[0] == name)  

	# subtract this name from allchar rdd
	allchar = allchar.subtractByKey(node0)


	# Find the nodes at a distance 1 and create (nodes1,1) 
	# where nodes1 are the nodes at a distance 1
	node_d = node0.flatMapValues(lambda x: x).map(lambda g: (g[1],1))

	# update allchar rdd by subtracting the characters just found
	allchar = allchar.subtractByKey(node_d)

	# This is the number of characters not touched by the search
	num_char_old = len(allchar.collect())


	# At this point we look for nodes at a distance 2
	d = 2


	# now we gonna iterate until d = 10
	while d<10:
         # Find the nodes at a distance d and creates (nodes,1)
         # where nodes are the nodes at a distance d
         node_d = nodes.join(node_d).map(lambda  x: (x[0], x[1][0] )).flatMapValues(lambda x: x).map(lambda g: (g[1],1)).groupByKey()
         node_d = node_d.map(lambda x: (x[0],1))
         
         # update allchar rdd by subtracting the characters just found
         allchar = allchar.subtractByKey(node_d)

         # This is the number of characters not touched by the search
         num_char_new = len(allchar.collect())
         
         

         if num_char_new < num_char_old:

         	# Continue the search 
         	d = d+1
         	num_char_old = num_char_new
         else:
         	# Stop the search
         	break

	return num_char_ini, num_char_new, d
