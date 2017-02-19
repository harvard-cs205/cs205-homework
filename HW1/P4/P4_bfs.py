#New version

def check_and_spawn((key,(adj_list,weight, remove)), counter):
	return_list = []
	if weight == counter:
		return_list.append((key,(adj_list,weight, 1)))    #set remove bit to 1 to remove it in the next iteration
		for neighbor in adj_list:
			return_list.append((neighbor,([], counter+1, -1)))
	else:
		return_list.append((key,(adj_list,weight, 0)))
	return return_list

def shrink_children((adj_list,weight, remove), (adj_list2,weight2, remove2) ,accum ):
	if (len(list(adj_list)) == 0):
		if weight2 == -1:
			accum+=1
			return (adj_list2, weight, remove2)
		else:
			return (adj_list2, weight2, remove2)
	else:
		if (weight == -1):
			accum+=1
			return (adj_list, weight2, remove)
		else:
			return (adj_list, weight, remove)


def bfs(rdd, starting_node, sc):

	#add a column for distance. initialize to 0 for starting node, and -1 for others
	#add a second column (remove) for whether this node should be filtered out. ie. has it been reached from the starting node
	bfs_rdd = rdd.map(lambda (x,y): (x,(y,0, 0)) if (x == starting_node) else (x,(y,-1, 0)) )

	#result dictionary for the number of nodes touched in each iteration
	results = {};
	results[0] = 1
	total_num = bfs_rdd.count() - 1

	#intialize accumulator as 1 to start the while loop
	accum = sc.accumulator(1)

	counter = 0
	while (accum.value != 0): 

		#we only need to work with those records that hasn't been reached yet
		filtered_rdd = bfs_rdd.filter(lambda (x,(y,z, remove)): remove == 0)
		child_expanded_rdd = filtered_rdd.flatMap(lambda (x,(y,z, remove)): check_and_spawn( (x,(y,z, remove)), counter))
		#print child_expanded_rdd.count()

		#reset accumulator
		accum = sc.accumulator(0)
	
		#reduce the extra records created by flatmap
		bfs_rdd = child_expanded_rdd.reduceByKey(lambda (y,z, remove1), (a,b, remove2): shrink_children((y,z, remove1),(a,b, remove2), accum))
		bfs_rdd = bfs_rdd.filter(lambda (x,(y,z, remove)): remove != -1)
		#print bfs_rdd.count()

		new_nodes_rdd = bfs_rdd.filter(lambda (x,(y,z, remove)): z == counter+1)
		new_nodes_num =  new_nodes_rdd.count()


		#tallying results
		if (new_nodes_num == 0):
			break
		results[counter+1] = new_nodes_num
		total_num = total_num - new_nodes_num

		counter+=1;

	results[-1] = total_num
	return results

