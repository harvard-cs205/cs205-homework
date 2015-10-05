import pyspark
sc = pyspark.SparkContext()

import time

n_parts = 100 # num partitions
links_fname  = "s3://Harvard-CS205/wikipedia/links-simple-sorted.txt"
titles_fname = "s3://Harvard-CS205/wikipedia/titles-sorted.txt"

pairings      = {}
pairings["0"] = {"source": "Kevin_Bacon", 		 "target": "Harvard_University"} 
pairings["1"] = {"source": "Harvard_University", "target": "Kevin_Bacon"	   } 

''' bonus pairings '''

pairings["2"] = {"source": "Kevin_Bacon",      "target": "Captain_America"} 
pairings["3"] = {"source": "Captain_America",  "target": "Kevin_Bacon"	  } 

pairings["4"] = {"source": "Captain_America", 		"target": "O_Captain!_My_Captain!"} 
pairings["5"] = {"source": "O_Captain!_My_Captain!", "target": "Captain_America"	  } 


def quiet_logs(sc):
	''' Shuts down log printouts during execution (thanks Ray!) '''
	logger = sc._jvm.org.apache.log4j
	logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
	logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
	logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)
	
quiet_logs(sc)

def update_distance(d):
	''' updates distance value for nodes when touched by BFS '''
	return lambda x: (x,d+1)

def mark_step(step): 
	''' keeps track of distance when reversing through BFS to get path '''
	return lambda x: (x[0],step)

def graph_to_chain(_):
	''' formats graph before union with master_chains, gives  (current_dist_node,previous_dist_node) edge '''
	''' note: nodes is in post-join() form: (previous_dist_node, (([associated_current_dist_nodes],None),0))
		where None==nodes[1][0][1] and 0==nodes[1][1] are remnants of the join, and we can ignore them.'''
	return lambda nodes: [(int(node),nodes[0]) for node in nodes[1][0][0]]

def set_target_node(d):
	''' formats target node for path-reversal sequence '''
	return lambda x: (x[0],d)

def make_rdds(tfn,lfn,n):
	''' creates title, graph RDDs '''
	titles = (sc.textFile(titles_fname,n)
				.zipWithIndex()
				.map(lambda x: (x[0],x[1]+1))
				.map(lambda x: tuple(reversed(x)))
				.partitionBy(n)
			)

	graph  = (sc.textFile(lfn,n_parts)
				 .cache()
				 .map(lambda x: x.split(" "))
				 .map(lambda x: (int(x[0][:-1]),(x[1:],None))) # [:-1] omits colon, gives (node,[associated]) tuples
				 .partitionBy(n)
			 )

	return titles,graph

titles,graph = make_rdds(titles_fname,links_fname,n_parts)

titles.cache()
graph.cache()


for pair in pairings.values():
	
	timeA = time.time() # benchmarking

	''' finder is a toggle: have we reached a BFS distance that contains our target? '''
	finder     = sc.accumulator(0) # start at one for single-source root
	counter    = sc.accumulator(0) # start at one for single-source root

	source = pair["source"]
	target = pair["target"]
	
	print "Find path:",source,"-->",target
	
	target_node = (titles.filter(lambda x: x[1]==target) # get target (index,title)
						 .map(lambda x: (x[0], 0)) 	 # convert to (index, 0)
						 .partitionBy(n_parts)			 
						 .cache()						 
				  )
	
	root_list   = (titles.filter(lambda x: x[1]==source) # get source (index,title)
						 .map(lambda x: (x[0], 0))	 # convert to (index, 0)
						 .partitionBy(n_parts)
						 .cache()
			 )

	master_root_list = root_list
	''' master_chains keeps a (target,source) tuple for every edge touched by BFS, starts with (root, None) '''
	master_chains    = root_list.map(lambda x: (x[0], None), preservesPartitioning=True)
	
	distance = counter.value

	while not finder.value: # iterate until target is found

		print "Distance:",distance

		graph2 = graph.join(root_list, numPartitions=n_parts).cache() # graph starting from current-distance root nodes

		master_chains = (master_chains.union(graph2.flatMap(graph_to_chain(None))
												   .distinct(numPartitions=n_parts)
											)
									  .partitionBy(n_parts)
						 )
		
		root_list = (graph2.flatMap( lambda x: x[1][0][0] )   # leaves only associated nodes at current distance
							.map( update_distance(distance) ) # assigns each node current distance value (node, dist)
							.partitionBy(n_parts)
							.leftOuterJoin( master_root_list, numPartitions=n_parts )
							.filter( lambda x: x[1][1] is None ) # combined with the LOJ, this is like a reverse join as x[1][1] is only none for nodes not in master root
							.map( lambda x: (int(x[0]),x[1][0])) # keeps (node,distance) to make up new root_list
							.distinct(numPartitions=n_parts)
						)
		master_root_list = (master_root_list.union( root_list )
										   .distinct(numPartitions=n_parts)
							) # adds new root_list to master
		counter.add(1)
		distance = counter.value
		root_list.join(target_node).foreach( lambda x: finder.add(1) )
	
	countback = sc.accumulator(distance)
	link_node = target_node
	path      = target_node.map(set_target_node(distance), preservesPartitioning=True)
	
	''' retrace our steps back to root node, picking a single node per distance to build a full path '''
	while countback.value > 1:
		link_node = (master_chains.join(link_node, numPartitions=n_parts)
								  .map(lambda x: (x[1][0],0))
								  .distinct(numPartitions=n_parts)
					)
		countback.add(-1)
		backstep = countback.value
		path = (path.union(link_node.map(mark_step(backstep)))
				    .distinct(numPartitions=n_parts)
				   )


	print "Report:",source,"-->",target
	print
	''' formatting: titles.join(path) gives (node, (node_name,node_distance)) nested tuples.
	    			so x[1][1] is node distance, and x[1][0] is node_name '''
	print (titles.join(path).map(lambda x: (x[1][1],x[1][0]))
							.partitionBy(n_parts)
							.groupByKey()
							.map(lambda x: list(x[1])[0])
							.collect()
			)


	timeB=time.time()        # benchmarking
	time_diff = timeB-timeA  # benchmarking

	print "Total time:", time_diff
