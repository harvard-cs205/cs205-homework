from pyspark import SparkContext
import time
#New version

def check_and_spawn((key,(adj_list,weight, parent)), counter):
	return_list = []
	return_list.append((key,(adj_list,weight, parent)))
	if weight == counter:
		for neighbor in adj_list:

			#remove self reference
			if (neighbor != key):
				return_list.append((neighbor,([], counter+1,key)))

	return return_list

def shrink_children((adj_list,weight, parent), (adj_list2,weight2, parent2) ,accum ):
	if (len(list(adj_list)) == 0):
		if weight2 == -1:
			accum+=1
			return (adj_list2, weight, parent)
		else:
			return (adj_list2, weight2, parent2)
	else:
		if (weight == -1):
			accum+=1
			return (adj_list, weight2, parent2)
		else:
			return (adj_list, weight, parent)


def bfs(rdd, starting_node, target_node, sc):

	#add a column for distance. initialize to 0 for starting node, and -1 for others
	bfs_rdd = rdd.map(lambda (x,y): (x,(y,0,-1)) if (x == starting_node) else (x,(y,-1,-1)) )

	#result dictionary for the number of nodes touched in each iteration
	results = [];

	#intialize accumulator as 1 to start the while loop
	accum = sc.accumulator(1)

	counter = 0
	while (accum.value != 0): 

		child_expanded_rdd = bfs_rdd.flatMap(lambda (x,(y,z, parent)): check_and_spawn( (x,(y,z, parent)), counter))

		#reset accumulator
		accum = sc.accumulator(0)
	
		#reduce the extra records created by flatmap
		bfs_rdd = child_expanded_rdd.reduceByKey(lambda (y,z, parent), (a,b, parent2): shrink_children((y,z, parent),(a,b, parent2), accum))
		#print bfs_rdd.count()

		new_nodes_rdd = bfs_rdd.filter(lambda (x,(y,z, parent)): z == counter+1)
		new_nodes_num =  new_nodes_rdd.count()


		#check if target node has been marked
		target_node_checker = bfs_rdd.lookup(target_node)
		print target_node_checker


		#Found target with none negative distance
		if (target_node_checker[0][1] != -1):
			break

		#if there are no more nodes to check return empty list
		if (new_nodes_num == 0):
			return []

		counter+=1;

	#recreate the path using the prev_node_master dict by tracing the steps backwards
	curr_node = target_node
	while (1):
		results.append (curr_node)
		print bfs_rdd.lookup(curr_node)
		curr_node = bfs_rdd.lookup(curr_node)[0][2]


		if curr_node == starting_node:
			break

	results.append (starting_node)
	print results
	return results






#----------------Path searching function----------------------------------------------

def find_path(start_name, target_name, sc):
	#initialize spark

	#get the links


	#get the raw data
	raw_data = "/home/zhiqian/Documents/cs205data/titles-sorted.txt"
	#raw_data = "s3://Harvard-CS205/wikipedia/titles-sorted.txt"
	page_names = sc.textFile(raw_data)

	links = sc.textFile("/home/zhiqian/Documents/cs205data/links-simple-sorted.txt")
	#links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')

	#process page_names. first, add an index
	page_names2 = page_names.zipWithIndex()
	#then flip the index to use it as a key
	page_names3 = page_names2.map(lambda (x,y): (y,x))

	#process the links by splitting the string up
	#key is the page, value is the list of children
	links2 = links.map(lambda x: x.split(": "))
	links3 = links2.map(lambda (x,y): (x,y.split(" ")))


	#look up for the starting node for the given links. the values of 1-indexed
	starting_node = page_names2.lookup(start_name)[0] + 1
	target_node = page_names2.lookup(target_name)[0] + 1
	print starting_node
	print target_node

	results_index = bfs(links3, str(starting_node), str(target_node), sc)
	print results_index

	results = []

	#using the index, search for the string name
	for ind in reversed(results_index):
		#minus 1 for the 1-index
		results.append( page_names3.lookup(int(ind) -1)[0] )



	print results


#--------------main------------------------------------

sc = SparkContext('local', "Wikipedia")



start2 = time.time()
find_path('Harvard_University', 'Kevin_Bacon', sc)
#find_path('Kevin_Bacon', 'Harvard_University', sc)
print time.time() - start2

