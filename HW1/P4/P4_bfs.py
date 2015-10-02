def init_start_node(x, v):
	if x[0] == v:
		return (x[0], ( 0, x[1][1]))
	else:
		return x
	
def bfs(g, v, max_iter):
#g is an RDD represent the graph
#each item in g is in the form (char name, [neighbor 1, neighbor 2, ...])

#v is the starting point of this BFS
	num_of_iter = 0
	num_node_not_touch_start = 0
	g = g.map(lambda x: (init_start_node(x, v ) ) )
	for i in range(max_iter):
		num_of_iter = num_of_iter + 1
		g = g.flatMap(lambda x: update(x))
		g = g.reduceByKey(lambda x,y : reconstruct(x,y))
		#num_node_not_touch = g.filter(lambda x: x[1][0] == 999999).count()
		#if num_node_not_touch == num_node_not_touch_start:
			#break
		#num_node_not_touch_start = num_node_not_touch
	return (g,num_of_iter)#.filter(lambda x: x[1][0] < 999999)

def update(x):
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
	if d < 999999:
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
	
