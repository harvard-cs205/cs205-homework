def BFS(start, rdd):
    visited = set()
    not_visited = set([start])
    iteration = 0
    nums = []
    #print not_visited
    while(len(not_visited)>0):
        iteration += 1
        all_paths = rdd.filter(lambda KV: KV[0] in not_visited).collect()
        visited |= not_visited
        children = set()
        for KV in all_paths:
            children |= set(KV[1])
        not_visited = children-visited

    return len(visited)-1, iteration
