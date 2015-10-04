# P5 method

def ssbfs(charname, directedlist):  # identical as P4
    sc = directedlist.context
    result = sc.parallelize([(charname, 0)])
    accum = sc.accumulator(0)
    while True:
        result = result.partitionBy(min(result.count()/128 + 1, 64))
        curvalue = accum.value
        waiting = result.filter(lambda kv: kv[1] == curvalue)
        count = waiting.count()
        if count == 0:
            break
        coming = waiting.join(directedlist)
        coming = coming.flatMap(lambda kv: [(v, curvalue+1) for v in kv[1][1]])
        result = result.union(coming.distinct())
        result = result.reduceByKey(lambda a,b: min(a,b))
        accum.add(1)
    return result

def shortestpath(nodestart, nodeend, directedlist):
    sc = directedlist.context
    directedlist.cache()
    visited = set([nodestart])  # visited nodes
    current = set([nodestart])  # nodes of the current level
    result = sc.parallelize([(nodestart, None)])  # track the path, child-parent pairs

    while True:
        if nodeend in current:  # if found
            path = [nodeend]  # trace back from nodeend
            while True:
                pathway = result.lookup(path[-1])[0]
                if pathway:
                    path.append(pathway)
                else:
                    break
            return len(path) - 1, path[::-1]

        if len(current) == 0:  # if no node in current level
            break  # end the bfs

        waitinglist = directedlist.filter(lambda kv: kv[0] in current)  # reach out from current level
        waitinglist = waitinglist.flatMap(lambda kv: [(v, kv[0]) for v in kv[1]])
        nextlist = waitinglist.filter(lambda kv: not kv[0] in visited)  # new nodes we found
        nextlist = nextlist.reduceByKey(lambda a, b: a)  # choose one parent is enough
        result = result.union(nextlist)  # update our result
        
        current = set(nextlist.keys().collect())  # update two sets for next iteration
        visited = visited.union(current)
    return -1, []  # if not found

def connectedcomp(undirlist):
    sc = undirlist.context
    waitlist = undirlist.map(lambda kv: kv)
    n_comp = 0

    while True:
        waitlist.cache()
        count = waitlist.count()  # if no nodes left
        print 'CURRENT LENGTH:', count
        if count == 0:
            return n_comp
        nodestart = waitlist.take(1)[0][0]  # choose one node
        waitlist = waitlist.partitionBy(min(count/128 + 1, 64)) 
        n_comp += 1
        subtract = ssbfs(nodestart, waitlist)  # find all connected nodes
        waitlist = waitlist.subtractByKey(subtract)  # remove them

