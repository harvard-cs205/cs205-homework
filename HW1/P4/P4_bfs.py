def bfs(root, edge_lookup, sc): 
	finished = sc.parallelize([(root, 0)])
	new_size = sc.accumulator(0)

	for i in range(10): 
		new_size.value = 0 

		finished_neighbors = finished.filter(lambda (K,V): K == i).join(edge_lookup).flatMap(lambda (K,V): V[1])

		finished_neighbors = finished_neighbors.distinct().map(lambda x: (x, i+1), preservesPartitioning=True)
		possible_neighbors = finished_neighbors.leftOuterJoin(finished)

		new_elements = possible_neighbors.filter(lambda (K,V): V[1] == None).map(
			lambda (K,V): (K, V[0]), preservesPartitioning=True)

		new_elements.foreach(lambda x: new_size.add(1))

		finished = finished.union(new_elements).cache()

		if new_size.value == 0: 
			break 
		print finished.count()

	print "Finished Set: "
	print finished.count()
	print "\n\n"

	return finished
