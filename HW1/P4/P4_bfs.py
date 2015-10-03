# P4 bfs method

def ssbfs(charname, charadjlist):
    sc = charadjlist.context
    result = sc.parallelize([(charname, 0)])  # the rdd that keep track with the visited notes
    accum = sc.accumulator(0)  # seemingly playful
    while True:
        curvalue = accum.value  # the depth
        waiting = result.filter(lambda kv: kv[1] == curvalue)  # the nodes of deepest level
        if waiting.count() == 0:  # all visited for the connected component
            break
        coming = waiting.join(charadjlist)  # go one step
        coming = coming.flatMap(lambda kv: [(v, curvalue+1) for v in kv[1][1]])
        result = result.union(coming.distinct())  # add to visited
        result = result.reduceByKey(lambda a,b: min(a,b))  # take the minimal length for each node
        accum.add(1)
    return result  # return all distances
