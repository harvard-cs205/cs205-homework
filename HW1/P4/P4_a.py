import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import P4_bfs

def swap_KV(KV):
	return (KV[1],KV[0])

def find_earliest(L):
	if L[0]==None and L[1]==None:
		return None
	else:
		return min(x for x in list(L) if x is not None)

def update_rank_list(rank_list,char_list):
#	merges old rank list RDD=(CHAR,RANK) with new found characters RDD=(NEW_CHAR,NEW_RANK)
# 	from the current search
#	 We do a join here, but the shuffle is pretty minimal, since everything should be
# 	copartitioned and the two RDDs are pretty small
#
	return rank_list.leftOuterJoin(char_list).mapValues(find_earliest)

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

if __name__ == "__main__":
	csv=sc.textFile("source.csv")
	num_partitions=8
	data_char=csv.map(lambda x:tuple(x.strip('"').split('","'))).distinct()
	data_comic=data_char.map(swap_KV)
	graph_edges=data_comic.join(data_comic)\
		.values() \
		.filter(lambda i:i[0]!=i[1]) \
		.distinct()\
		.partitionBy(num_partitions).cache()

	info=''
	start_char=['CAPTAIN AMERICA','MISS THING/MARY','ORWELL']
	for char in start_char:
#		graph=P4_bfs.num_connections(graph_edges,char,num_partitions,sc)


#sets up rank - -1 means unprocessed, above that, the number indicates the ranks
	level=0
	rank_list=data_char.keys().distinct()\
		.map(lambda i:(i,None)).partitionBy(num_partitions).cache()
	
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

	


#	tot_char=data_char.keys().distinct().count()	
#	num_nodes=graph.count()
# 	rem=tot_char-num_nodes
# 	info=info+char+' has '+str(num_nodes)+' connected nodes.\n'+ str(rem)+\
# 		' of the characters are not connected. \n\n'
# 	
# 	with open('P4.txt','w') as f:
# 		f.write(info)	