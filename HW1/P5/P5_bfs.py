def swap_KV(KV):
	return (KV[1],KV[0])

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def find_earliest(L):
	if all(x==None for x in L):
		return None
	else:
		L_clean=tuple(x for x in L if x!=None)[0]
		L_level=tuple(x[0] for x in L_clean)
		return tuple(x for x in L_clean if x[0]==min(L_level))

def get_rank(i,level):
	if i[1]==None:
		return False
	else:
		return i[1][0][0]==level
	
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
#	assert copartitioned(rank_list,new_char_list) #just to be sure
	return rank_list.leftOuterJoin(new_char_list).mapValues(find_earliest)

def find_paths(graph_edges,start_page,end_page,page_names,numPartitions,sc):

############ Main BFS search Code##############
###############################################
#char=character to start search
# graph_edges= RDD of (K,V) where node K links to node V. Does not assume symmetric
# graph.
# numPartitions=number of partitions used
# sc=spark context

###############################################
	#find the index of start and finish page
	start_idx=page_names.filter(lambda (K,V):V==start_page).keys().collect()[0]
	end_idx=page_names.filter(lambda (K,V):V==end_page).keys().collect()[0]


	#set up rank
	# Rank RDD is of the form 
	#		[(NodeA,(LevelA,(Node1,NodeA2a,...)),(LevelA,(Node1,NodeA2b,...)) \
	#		(NodeB, (LevelB,(Node1,NodeB2,...)))]
	# Here, Node1 is the starting node, and the following nodes indicate
	# the path to get from the starting node to whichever node in question. 
	# In the example case of NodeA, there are multiple ways to get from Node1 to
	# NodeA in LevelA steps. LevelA is the minimum number of step required to 
	# get from Node1 to NodeA. If Node has not yet been touched, it has the 
	# value (None,None)
	
	rank_list=graph_edges.keys().distinct()\
		.map(lambda i:(i,None),preservesPartitioning=True)
	
	if not copartitioned(rank_list,graph_edges):
		rank_list=rank_list.partitionBy(numPartitions).cache()
	else:
		rank_list=rank_list.cache()
		
		
	level=0
	graph_init=(start_idx,((level,(start_idx,)),))

	rank_list=rank_list.map(lambda i:graph_init if i[0]==start_idx else i,\
		preservesPartitioning=True)
	
	Nmax=1
	graph_size=[1]
	char_paths=[]
	###################################
	#start search
#	for level in range(Nmax):
	while graph_size[level]!=0 and char_paths==[]:
		#reinitialize accumulator
		accum = sc.accumulator(0)
		
		#filter elements from previous level for search
		graph_level=rank_list.filter(lambda i:get_rank(i,level))
		
		assert copartitioned(graph_level,graph_edges)
		
		#join with the edges -- to find the connected nodes
		graph_level_join=graph_level.join(graph_edges)
		
		 #1. take the join, make the value the new key
		 #2. repartition by has of the new key		
		 #3. get rid of duplicates
		joined_char=graph_level_join\
			.map(lambda (K,V):(V[-1],tuple((level+1,x[1]+(V[-1],)) for x in V[0])))\
			.partitionBy(numPartitions)
		
		new_found_char=joined_char.reduceByKey(lambda a,b:a+b)
		
	#Merge newly connected characters with the master rank list
	#rank_list is the only information passed from one iteration to next, so cache it
		assert copartitioned(rank_list,new_found_char)

		rank_list=update_rank_list(rank_list,new_found_char).cache()
		new_char_list=rank_list.filter(lambda i:get_rank(i,level+1))
	# the rank list update will filter out characters that had already been found in
	# previous iterations, so now we need to count the filtered list
		new_char_list.foreach(lambda x: accum.add(1))
		#add to accumulator
		graph_size.append(accum.value)
		
		print(graph_size)
		char_paths_rdd=rank_list.filter(lambda (K,V):K==end_idx)\
			.values().flatMap(lambda i:i).values()
		char_paths=char_paths_rdd.collect()
		paths=[]
		for p in char_paths:
			paths.append(tuple(page_names.lookup(x)[0] for x in p))
		
		level+=1
	return (graph_size,paths)