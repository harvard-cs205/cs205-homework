import pyspark

# num partitions (4 times the number of cores)
N = 256

def shortest_paths(start, goal, graph):
	visited = sc.parallelize([(start, 0)], N)
	queue = sc.parallelize([(start, (start,))], N)
	paths = sc.emptyRDD()
	i = 0
	while paths.isEmpty():
		i += 1
		print i
		temp = queue
		joined = graph.join(queue, N).distinct()
		queue = joined.values().distinct().partitionBy(N)
		graph = graph.subtractByKey(temp, N).cache()
		queue = queue.map(lambda (u,v): (u, v+(u,)), preservesPartitioning=True)
		queue = queue.subtractByKey(visited, N).cache()
		visited = visited.union(queue).partitionBy(N).cache()
		paths = queue.values().filter(lambda x: x[-1] == goal).cache()
	return paths.collect()

def connected_components(root, graph):
	# initialize queue and distances rdds
	components = 0
	biggest_component = 0
	while not graph.isEmpty():
		components += 1
		if components != 1:
			root = graph.first()[0]
		distances = queue = sc.parallelize([(root, 0)], N)
		acc = sc.accumulator(1)
		iteration = 0
		while acc.value != 0:
			acc = sc.accumulator(0)
			iteration += 1
			# keep a temporary copy of the queue to later on remove items in queue from graph
			temp = queue
			# get all (neighbor, distance) value pairs for each key in queue
			joined = graph.join(queue, N).distinct()
			# have to partition again when taking the values
			queue = joined.values().distinct().partitionBy(N)
			# take out original queue keys from graph and cache for later iterations
			graph = graph.subtractByKey(temp, N).cache()
			# set the distance values to current iteration value
			newDistances = queue.mapValues(lambda _: iteration)
			# remove already visited keys from the queue, and cache for later iterations
			queue = newDistances.subtractByKey(distances, N).cache()
			# add keys in queue with their distances to distances rdd, and cache for later iterations
			distances = distances.union(queue).partitionBy(N).cache()
			# increment accumulator to satisfy the while condition (unless the queue is empty, in which case the loop stops after this)
			queue.foreach(lambda _: acc.add(1))
		size = distances.count()
		print size
		biggest_component = max(biggest_component, size)
	return components, biggest_component

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

def get_names_from_ids(paths, page_names):
	result = []
	id_dct = dict([(page_id, 1) for path in paths for page_id in path])
	name_dct = dict(page_names.filter(lambda (u,v): u in id_dct).collect())
	for i in range(len(paths)):
		pathi_names = []
		for j in range(len(paths[i])):
			pathi_names.append(name_dct[paths[i][j]])
		result.append(pathi_names)
	return result


sc = pyspark.SparkContext()

# make pyspark shut up
sc.setLogLevel('WARN')

# a lot of this code is taken from Professor's review session
links = links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()

neighbor_graph = links.map(link_string_to_KV)
neighbor_graph = neighbor_graph.flatMapValues(lambda v: v)
graph = neighbor_graph.partitionBy(256).cache()

# find Kevin Bacon
Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
# This should be [(node_id, 'Kevin_Bacon')]
assert len(Kevin_Bacon) == 1
Kevin_Id = Kevin_Bacon[0][0]  # extract node id

Harvard_University = page_names.filter(lambda (K, V): V == 'Harvard_University').collect()
# This should be [(node_id, 'Harvard_University')]
assert len(Harvard_University) == 1
Harvard_Id = Harvard_University[0][0]  # extract node id


shortest_paths_from_Kevin_to_Harvard = shortest_paths(Kevin_Id, Harvard_Id, graph)
print get_names_from_ids(shortest_paths_from_Kevin_to_Harvard, page_names)
shortest_paths_from_Harvard_to_Kevin = shortest_paths(Harvard_Id, Kevin_Id, graph)
print get_names_from_ids(shortest_paths_from_Harvard_to_Kevin, page_names)



# num_connected_components = connected_components(root, graph)
# print num_connected_components



