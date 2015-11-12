from pyspark import SparkContext, SparkConf

if 'sc' not in globals():
	conf = SparkConf().setAppName('BFS')
	sc = SparkContext(conf=conf)


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

def connected_components(graph_edges,full_char_list,numPartitions,sc):

############ Main BFS search Code##############
###############################################
# graph_edges= RDD of (K,V) where node K links to node V. Does not assume symmetric
# graph.
# char=RDD of distinct node values
# char and graph_edges should be copartitioned
# numPartitions=number of partitions used
# sc=spark context

###############################################
	char_list=graph_edges.groupByKey().keys()
	#because there are a number of pages that are unlinked in this dataset
	# we use the edges to determine which pages to search
	# that way, we know that every page we search has at least one connection
	
	rank_list=char_list.map(lambda i:(i,None),preservesPartitioning=True)
	
	if not copartitioned(rank_list,graph_edges):
		rank_list=rank_list.partitionBy(numPartitions).cache()
	else:
		rank_list=rank_list.cache()
	#as mentioned above, not there are a lot of pages in the full char list
	#that do not have any connections. Each of those is one connected component	
	num_connected_components=full_char_list.count()-char_list.count()
	while rank_list.count()!=0:	
		char=rank_list.keys().first() #take first char from rank_list
		print(char)		
		level=0	
		graph_init=sc.parallelize([(char,level)]).partitionBy(numPartitions)
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
				.partitionBy(numPartitions)
		
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
			print(graph_size)
			level+=1
		rank_list=rank_list.filter(lambda (K,V):V==None).cache()
		print('Number of Nodes Left= '+str(rank_list.count())) 
		num_connected_components+=1
	return num_connected_components
	################################################################################
################################################################################
################################################################################
################################################################################

	
if __name__ == "__main__":
	numPartitions=128		
	sc.setLogLevel("ERROR")
	links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',numPartitions)
	page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt',numPartitions)
	page_names = page_names.zipWithIndex().map(lambda (K,V):(V+1,K))\
		.partitionBy(numPartitions)
	graph_edges=links.map(lambda i:tuple(i.split(': '))) \
		.mapValues(lambda i:i.split(' ')) \
		.flatMapValues(lambda i:i).map(lambda i:(int(i[0]),int(i[1])))\
		.partitionBy(numPartitions)
	pages=page_names.keys()	 

	#make graph_edges with swapped K,V. Copartition with graph_edges
	swapped_graph_edges=graph_edges.map(swap_KV).partitionBy(numPartitions)
	merged_graph_edges=graph_edges+swapped_graph_edges
	
	#take combined graph_edges and swapped graph edges, group by key,
	# set gets rid of duplicates, then we reflatten. 
	#This is the graph assuming full symmetry.
	graph_edges_symmetric=merged_graph_edges.groupByKey()\
		.mapValues(lambda i:tuple(set(i))).flatMapValues(lambda i:i).cache()
		
	
	graph_edges_directed=merged_graph_edges.groupByKey()\
		.mapValues(lambda X:tuple(set([x for x in X if X.count(x) > 1])))\
		.flatMapValues(lambda i:i).cache()
	
	CC_sym=connected_components(graph_edges_symmetric,pages,numPartitions,sc)
	info='Number of connected components assuming symmetric graph: '+str(CC_sym)
  	with open('P5_connected_components.txt','a') as f:
   		f.write(info)
  	
 	CC_dir=connected_components(graph_edges_directed,pages,numPartitions,sc)
	info='Number of connected components neglecting directed edges: '+str(CC_sym)
	with open('P5_connected_components.txt','a') as f:
 		f.write(info)
