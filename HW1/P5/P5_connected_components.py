from pyspark import SparkContext
import time
import os

f1=open('./results.txt', 'w+')

#New version

def check_and_spawn((key,(adj_list,weight, remove)), counter):
	return_list = []

	if weight == counter:
		return_list.append((key,(adj_list,weight,1)))
		for neighbor in adj_list:
			#remove self reference
			if (neighbor != key):
				return_list.append((neighbor,([], counter+1, -1)))
	else:
		return_list.append((key,(adj_list,weight, 0)))
	return return_list

def shrink_children((adj_list,weight, remove), (adj_list2,weight2,remove2) ):
	if (len(list(adj_list)) == 0):
		#accumulator here?


		if weight2 == -1:
			#accum+=1
			return (adj_list2, weight,remove2)
		else:
			return (adj_list2, weight2, remove2)
	else:
		if (weight == -1):
			#accum+=1
			return (adj_list, weight2, remove)
		else:
			return (adj_list, weight, remove)


def bfs(rdd, sc):

	#add a column for distance. initialize to -1. second column is for starting nodes
	bfs_rdd = rdd.map(lambda (x,y): (x,(y,-1, 0)))
	results = []
	total = bfs_rdd.count()
	#print total

	#disable accumulators as it crashes the system
	while (total > 0):

		bfs_rdd = bfs_rdd.filter(lambda (x,(y,z, remove)): remove == 0)
		#bfs_rdd = bfs_rdd.filter(lambda (x,(y,z, remove, parent)): parent == -1)
		starting_node = bfs_rdd.first()[0]

		#print starting_node
		current_count = 1
		total = total-1
		#print starting_node
		bfs_rdd = bfs_rdd.map(lambda (x,(y,z, remove)): (x,(y,0, remove)) if (x == starting_node) else (x,(y,z,remove)) )		
		#intialize accumulator as 1 to start the while loop
		#accum = sc.accumulator(1)

		counter = 0
		#while (accum.value != 0): 
		while(1):
			#we only need to work with those records with weight = counter, or untouched yet
			bfs_rdd = bfs_rdd.filter(lambda (x,(y,z, remove)): remove == 0).cache()
			bfs_rdd = bfs_rdd.flatMap(lambda (x,(y,z, remove)): check_and_spawn( (x,(y,z, remove)), counter))
			#print child_expanded_rdd.count()
			#print starting_node
			#reset accumulator
			#accum = sc.accumulator(0)
		
			#reduce the extra records created by flatmap
			bfs_rdd = bfs_rdd.reduceByKey(lambda (y,z, remove1), (a,b, remove2): shrink_children((y,z,remove1),(a,b, remove2)))
			bfs_rdd = bfs_rdd.filter(lambda (x,(y,z, remove)): remove != -1)
			#print bfs_rdd.count() 

			new_nodes_rdd = bfs_rdd.filter(lambda (x,(y,z, remove)): remove != -1 and z == counter+1)
			new_nodes_num =  new_nodes_rdd.count()
			#print bfs_rdd.collect()
			#print new_nodes_rdd.collect()
			#if there are no more nodes to check return empty list
			if (new_nodes_num == 0):
				break

			total = total - new_nodes_num
			current_count = current_count + new_nodes_num
			counter+=1;

		print str(starting_node) + "   " + str(current_count) + "     " +str(total) + "    "+ str(counter)
		print >> f1, str(starting_node) + "   " + str(current_count) + "     " + str(total)+ "   "+ str(counter)

		#print
		#print total
		#print bfs_rdd.collect()
		#print
		results.append(current_count)
		#print bfs_rdd.collect()
	return results


def split_list((key,list_of_neighbors)):
	return_list = []

	for neighbor in list_of_neighbors:
		return_list.append(((key, neighbor), 1))
		return_list.append(((neighbor,key), 1))
	return return_list	



#initialize spark
sc = SparkContext('local', "Wikipedia")

#get the links

start = time.time()


#toggle this for running on AWS
links = sc.textFile("/home/zhiqian/Documents/cs205data/links-simple-sorted.txt", 32)
#links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 128)


#process the links by splitting the string up
#key is the page, value is the list of children
links2 = links.map(lambda x: x.split(": "))
links3 = links2.map(lambda (x,y): (x,y.split(" ")))



links4 = links3.flatMap(split_list)
links5 = links4.reduceByKey(lambda x, y: x+y)


#Toggle this for converting bidirected edges to a single edge
#links5 = links5.filter (lambda (x,z): z > 1)


rdd_intersect2 = links5.map(lambda ((x,y), z): (x,y))
rdd_intersect2 = rdd_intersect2.groupByKey().mapValues(lambda x: list(x)).cache()

print rdd_intersect2.count()
print >> f1, rdd_intersect2.count()
print time.time() - start
print >> f1, time.time() - start


results_index = bfs(rdd_intersect2, sc)

print results_index
print >> f1, results_index

print max(results_index)
print >> f1, max(results_index)

print time.time() - start
print >> f1, time.time() - start

