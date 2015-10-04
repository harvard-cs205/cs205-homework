def init_start_node(x, v):
	if x[0] == v:
		return (x[0], ( 0, x[1][1]))
	else:
		return x
	
def bfs(g, v, max_iter, sc):
#g is an RDD represent the graph
#each item in g is in the form (char name, [neighbor 1, neighbor 2, ...])
	num_of_node = sc.accumulator(0)
#v is the starting point of this BFS
	num_of_iter = 0
	num_node_not_touch_start = 0
	g = g.map(lambda x: (init_start_node(x, v ) ) )
	for i in range(max_iter):
		#num_node_already_touch = num_of_node
		num_of_iter = num_of_iter + 1
		g = g.flatMap(lambda x: update(x, num_of_iter - 1))
		g = g.reduceByKey(lambda x,y : reconstruct(x,y, num_of_node))
		num_new_touch = g.filter(lambda x: x[1][0] == num_of_iter).count()
		if num_new_touch == 0:
			break
	return (g, num_of_iter, num_of_node.value)#.filter(lambda x: x[1][0] < 999999)

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
			result.append((i, (d + 1, [])))
	return result


def reconstruct(x, y, accum):
# reduceBykey function, find the shortest distance of each node and combine its parents
# x y is the value of a same key in the form (distance, [list of parents])
	d = min(x[0], y[0])
	if(x[0] >= 999999 or y[0] >= 999999):
		accum.add(1)
	a_list = list(set(x[1] + y[1]))
	return (d, a_list)
	
