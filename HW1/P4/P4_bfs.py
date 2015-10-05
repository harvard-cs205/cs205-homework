def swap_KV(KV):
	return (KV[1],KV[0])

def find_earliest(L):
	if L[0]==None and L[1]==None:
		return None
	else:
		return min(x for x in list(L) if x is not None)

def update_rank_list(rank_list,new_char_list):
	#	merges old rank list RDD=(CHAR,RANK) 
	#	with new found characters RDD=(NEW_CHAR,NEW_RANK)
	# 	from the current search
	#	We do a join here, but the shuffle is pretty minimal, 
	#	since everything should be
	# 	copartitioned and the two RDDs are pretty small
	
	#	The left outer join will return a value of (x,y) where x is the old rank
	#	and y is the new rank. If node has not been touched yet, x=None
	#	If node is touched at this level, y=level
	#	The mapValues function filters (x,y) to return either None, if 
	#	node hasn't been touched even after this search level,
	#or the lowest level found thus far
	assert copartitioned(rank_list,new_char_list) #just to be sure
	return rank_list.leftOuterJoin(new_char_list).mapValues(find_earliest)

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def num_connections_bfs(char,graph_edges,char_list,num_partitions,sc):

############ Main BFS search Code##############
###############################################
#char=character to start search
# graph_edges= RDD of (K,V) where node K links to node V. Does not assume symmetric
# graph.
# char_list=RDD of distinct node values
# char_list and graph_edges should be copartitioned
# num_partitions=number of partitions used
# sc=spark context

###############################################

	#set up rank - None means unprocessed, above that, the number indicates the ranks
	rank_list=char_list.map(lambda i:(i,None),preservesPartitioning=True)
	
	if not copartitioned(rank_list,graph_edges):
		rank_list=rank_list.partitionBy(num_partitions).cache()
	else:
		rank_list=rank_list.cache()
		
		
	level=0	
	graph_init=sc.parallelize([(char,level)]).partitionBy(num_partitions)

	rank_list=update_rank_list(rank_list,graph_init)	
	
	Nmax=3
	graph_size=[1]
	###################################
	#start search
	while graph_size[level]!=0:
		#reinitialize accumulator
		accum = sc.accumulator(0)
		
		#filter elements from previous level for search
		graph_level=rank_list.filter(lambda i:i[1]==level)
		
		assert copartitioned(graph_level,graph_edges)
		
		#join with the edges -- to find the connected nodes
		graph_level_join=graph_level.join(graph_edges)
		
		
		
		 #1. take the join, make the value the new key
		 #2. repartition by has of the new key		
		 #3. get rid of duplicates
		joined_char=graph_level_join.map(lambda i:(i[1][1],level+1))\
			.partitionBy(num_partitions)
		
		new_found_char=joined_char.groupByKey().mapValues(lambda i:min(list(i)))	
		
	#Merge newly connected characters with the master rank list
	#rank_list is the only information passed from one iteration to next, so cache it
		assert copartitioned(rank_list,new_found_char)
		
		rank_list=update_rank_list(rank_list,new_found_char).cache()
		
		
	# the rank list update will filter out characters that had already been found in
	# previous iterations, so now we need to count the filtered list
		new_char_list=rank_list.filter(lambda (K,V):V==level+1)
		new_char_list.foreach(lambda x: accum.add(1))
		
		#add to accumulator
		graph_size.append(accum.value)
		level+=1
	return graph_size