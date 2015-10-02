def bfs(char_pairs, start_char, sc):
    """
    Given a graph RDD and a starting character name, performs a breadth-first search and
    returns an RDD of all nodes encountered.
    """
    
    def start_dist(k,v):
    	" Changes the distance for the starting character to zero. "
    	if k == start_char:
        	return (k,0)
    	else:
        	return (k,v)
    
    def update_dist(v):
		" Updates nodes' distance from the starting node and uses an accumulator to check if any distances are updated. "
		if v[1] < 0:
			accum.add(1)
			return (v[0], counter)
		return v
        
    accum = sc.accumulator(0)
    dist = char_pairs.keys().distinct().map(lambda v: (v,-1))
    dist = dist.map(lambda (k,v): start_dist(k,v))
    dist = dist.partitionBy(20)
    connected = sc.parallelize([(start_char, None)])
    connected = connected.partitionBy(20)
    operations = 0
    counter = 1

    # Perform BFS
    while True:
        connected = char_pairs.join(connected.mapValues(lambda v: None)).mapValues(
            lambda v: (max(v), max(v))).values().distinct().partitionBy(20)
        new_dists = connected.join(dist.filter(lambda (k,v): v<0)).mapValues(update_dist).values().partitionBy(20)
        new_dists.count() # Force computation to update accumulator

        if not accum.value > operations:
            break
        operations = accum.value
        counter += 1

        dist = new_dists.union(dist).reduceByKey(lambda x,y: max(x,y))

    return dist.filter(lambda (k,v): v>=0)