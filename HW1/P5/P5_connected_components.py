#import pyspark
#import time
import time
import findspark
findspark.init('/home/zelong/spark')
import pyspark
sc = pyspark.SparkContext()

#links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
#page_names = sc.textFile("s3://Harvard-CS205/wikipedia/titles-sorted.txt") 
#name_index = page_names.zipWithIndex().map(lambda (x,y): (x, y+1 ))
#key_index = name_index.map(lambda (x,y): (y,x))
#print name_index.take(5)

########################################################################################
#Construct the graph 
def link_to_KV(s):
    parent, children = s.split(': ')
    children = [int(to) for to in children.split(' ')]
    return (int(parent),( 999999, children))
links = sc.textFile("links-simple-sorted.txt")
g = links.map(lambda x : link_to_KV(x))

#make all the link symmetric 
def add_parent(x):
#x is a key-value rdd item in the form (node, (distance, [list of neighbor]))
#update the x's neighbor's distance to x.dist + 1
# return an key-value rdd with item in the form of
# (one_neighbor_of_x, (distance+1, [list of parent]))
#x is the parent of all its neighbor a,b,c,d...
#a,b,c,d,e can also be the parent of x
	children = x[1][1]
	curr = x[0]
	result = [x]
	#check the distance of x, if not infinite
	
		#update the distance of x's neighbor  
	for i in children:
		result.append((i, (999999, [curr])))
	return result
def update_parent(x,y):
	a_list = list(set(x[1] + y[1]))
	return (999999, a_list)

g=g.flatMap(lambda x: add_parent(x)).reduceByKey(lambda x,y: update_parent(x,y))
g = g.cache()
########################################################################################
#Define BFS function

def bfs(g, v):
#g is an RDD represent the graph
#each item in g is in the form (char name, [neighbor 1, neighbor 2, ...])
#v is the starting point of this BFS
	level = 0
	num_node_touch = 1
	g = g.map(lambda x: (init_start_node(x, v ) ) )
	while True:
		#num_node_already_touch = num_of_node
		level = level+1
		g = g.flatMap(lambda x: update(x, level - 1))
		g = g.reduceByKey(lambda x,y : reconstruct(x,y))
		num_new_touch = g.filter(lambda x: x[1][0] == level).count()
		num_node_touch = num_node_touch + num_new_touch
		if num_new_touch == 0:
			break
	return (g, level, num_node_touch)

#helper function for bfs
def init_start_node(x, v):
	if x[0] == v:
		return (x[0], ( 0, x[1][1]))
	else:
		return x

def update(x, level):
#x is a key-value rdd item in the form (node, (distance, [list of neighbor]))
#update the x's neighbor's distance to x.dist + 1
# return an key-value rdd with item in the form of
# (one_neighbor_of_x, (distance+1, [list of parent]))
#x is the parent of all its neighbor a,b,c,d...
#a,b,c,d,e can also be the parent of x
	d = x[1][0]
	neighbor = x[1][1]
	curr = x[0]
	result = [x]
	#check the distance of x, if not infinite
	if d == level:
		#update the distance of x's neighbor  
		for i in neighbor:
			# i is the name of one neighbor
			result.append((i, (d + 1, [curr])))
	return result



def reconstruct(x, y):
# reduceBykey function, find the shortest distance of each node and combine its parents
# x y is the value of a same key in the form (distance, [list of parents])
	d = min(x[0], y[0])
	a_list = list(set(x[1] + y[1]))
	return (d, a_list)
#######################################################################################
#define get_path function and generate txt file

starttime = time.time()

def get_connected(g):
	connect_list = []
	i = 0
	while True:
		i = i+1
		start_node = g.take(1)[0][0]
		result = bfs(g, start_node)
		new_g = result[0]
		level = result[1]
		num_node = result[2]
		g = new_g.filter(lambda x: x[1][0] > level + 2)
		if g.count() == 1:
			connect_list.append((i+1, 1))	
			break
		connect_list.append((i, num_node))
	return connect_list

print get_connected(g)
print time.time() - starttime
