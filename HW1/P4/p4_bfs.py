def BFS(graph, source_node):
    # set accumulator to 0
    accum = sc.accumulator(0)
    accum.value=0
    chars_list = sc.parallelize([(source_node,0)])
    new_chars = sc.parallelize([source_node])
    # while no new characters to add to list
    while not new_chars.isEmpty():
        level=accum.value
	# new_chars is a flatmap of all children of all of the nodes at the current value
        new_chars=chars_list.filter(lambda x: x[1]==level).join(graph).map(lambda x:[(a,x[1][0]+1) for a in x[1][1]]).flatMap(lambda x:x).distinct()
        accum.add(1)
        # union the new characters with the current list, and only get the one with the lowest value if there are duplicates
	chars_list=chars_list.union(new_chars).groupByKey().mapValues(lambda x: min(x))
    return chars_list