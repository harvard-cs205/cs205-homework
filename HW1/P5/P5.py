import findspark
findspark.init("/home/zelong/spark")
import pyspark
import time

sc = pyspark.SparkContext()

links = sc.textFile("links-simple-sorted.txt") 
page_names = sc.textFile("titles-sorted.txt") 
name_index = page_names.zipWithIndex().map(lambda (x,y): (x, y+1 ))
key_index = name_index.map(lambda (x,y): (y,x))
#print name_index.take(5)

########################################################################################
#Construct the graph 
def link_to_KV(s):
    parent, children = s.split(': ')
    children = [int(to) for to in children.split(' ')]
    return (int(parent),( 999999, int(parent), children))

g = links.map(lambda x : link_to_KV(x))

########################################################################################
#Define BFS function

def bfs(g, v, end_v , max_iter):
#g is an RDD represent the graph
#each item in g is in the form (char name, [neighbor 1, neighbor 2, ...])
	#num_of_node = sc.accumulator(0)
#v is the starting point of this BFS
	num_of_iter = 0
	num_node_not_touch_start = 0
	g = g.map(lambda x: (init_start_node(x, v ) ) )
	flag = True
	for i in range(max_iter):
		#num_node_already_touch = num_of_node
		num_of_iter = num_of_iter + 1
		g = g.flatMap(lambda x: update(x, num_of_iter - 1, end_v))
		g = g.reduceByKey(lambda x,y : reconstruct(x,y))
		#num_new_touch = g.filter(lambda x: x[1][0] == num_of_iter).count()
		dist_target = g.lookup(end_v)[0][0] 
		if dist_target < 999999:
			break
		#if num_new_touch == 0:
		#	break
		#if flag:
		#	break
	return (g, num_of_iter)#.filter(lambda x: x[1][0] < 999999)

#helper function for bfs
def init_start_node(x, v):
	if x[0] == v:
		return (x[0], ( 0,x[0], x[1][2]))
	else:
		return x

def update(x, level, end_v):
#x is a key-value rdd item in the form (node, (distance, [list of neighbor]))
#update the x's neighbor's distance to x.dist + 1
# return an key-value rdd with item in the form of
# (one_neighbor_of_x, (distance+1, [list of parent]))
#x is the parent of all its neighbor a,b,c,d...
#a,b,c,d,e can also be the parent of x
	d = x[1][0]
	children = x[1][2]
	parent = x[1][1]
	curr = x[0]
	result = [x]
	#check the distance of x, if not infinite
	if d == level:
		#update the distance of x's neighbor  
		for i in children:
			#if i == end_v:
				#flag = True
			# i is the name of one neighbor
			result.append((i, (d + 1, curr , [])))
	return result


def reconstruct(x, y):
# reduceBykey function, find the shortest distance of each node and combine its parents
# x y is the value of a same key in the form (distance, [list of parents])
	if x[0] < y[0]:
		d = x[0]
		parent = x[1]
	else:
		d = y[0]
		parent = y[1]
	a_list = list(set(x[2] + y[2]))
	return (d, parent, a_list)

#######################################################################################
#define get_path function and generate txt file

starttime = time.time()

#function call the bfs function
#generate path 
#write to txt file
f = open('P5.txt','w')
def get_path(g,name,key, start_name, end_name):
	start = name_index.lookup(start_name)[0]
	end = name_index.lookup(end_name)[0]
	result_rdd = bfs(g, start, end, 10)[0]
	result = [end]
	dest_value = result_rdd.lookup(end)[0]
	distance = dest_value[0]
	target_parent = dest_value[1]
	for i in range(distance):
		result.append(target_parent)
		temp = result_rdd.lookup(target_parent)[0][1]
		target_parent = temp
	result_name = []
	for j in reversed(result):
		temp = key.lookup(j)[0]
		result_name.append(temp)
	#print distance, result_name
	summary = "From %s to %s, the distance is %d\n" % (start_name,end_name,distance)
	f.write(summary)
	path = "The path is: "
	for i in result_name:
		path = path + i + "--->"
	f.write('\n')
	f.write(path)

get_path(g, name_index, key_index,"Harvard_University","Kevin_Bacon" )
get_path(g, name_index, key_index,"Kevin_Bacon","Harvard_University")
f.close()
print time.time() - starttime

