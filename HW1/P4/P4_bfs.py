import sys
import functools


# put the whole process into a function, given an rdd which holds as elements
# (v, [v_1, ..., v_d]),  a start node v0 and the spark context corresponding to the rdd
def sparkBFS(context, rdd, v0):
    
    # prepare data structure for single source shortest paths
    imaxvalue = sys.maxint

    # currently our rdd looks like (v, [v1, v2, ...]). now we map it to a tuple with
    # (v, (<adj. list.>, disttov0, color)) where disttov0 is 0 for v = v0 and imaxvalue else,
    # color = GRAY for v0 and BLACK else
    # WHITE means vertex not visited yet
    # GRAY means vertex is visited in the next hop
    # BLACK mean vertex already visited

    # to speedup we use
    # WHITE = 2
    # GRAY = 1
    # BLACK = 0
    rdd = rdd.map(lambda x: (x[0], x[1], 0 if x[0] == v0 else imaxvalue, 1 if x[0] == v0 else 2))
    
    # map to (K, V) form
    rdd = rdd.map(lambda x: (x[0], (x[1], x[2], x[3])))
    
    # helper functions for one hop START
    
    # say we have given (v, ([v_1, ..., v_d], d, 'GRAY'))
    # this will be expanded to 
    # (v_1, (NULL, d+1, 'GRAY'))
    # ...
    # (v_d, (NULL, d+1, 'GRAY'))
    # (v, ([v_1, ..., v_d], d, 'BLACK'))
    # in the next step we can then call a reducebykey to update distances/adjacency lists
    def expandNode(x):
        if x[1][2] == 1: # 'GRAY'
        # set current node to visited
            res = []
            res.append( (x[0], (x[1][0], x[1][1], 0)) ) # 'BLACK'

            # spawn new GRAY nodes
            for i in range(0, len(x[1][0])):
                res.append( (x[1][0][i], ([], x[1][1] + 1, 1)) ) # 'GRAY'

            return tuple(res)
        else: 
            return [x]
        
    # in the next step we combine all tuples for the same key returning
    # the minimum distance, longest adjacency list and darkest color
    # the algorithm will determine if there is no gray node left
    def reduceNodes(a, b):
        res = (a[0] + b[0], min(a[1], b[1]), min(a[2], b[2]))

        # return a tuple of 3 entries
        return res
    
    def countGrayNodes(x, gray_accum):
        # inc count of remaining gray nodes by 1!
        if x[1][2] == 1:
            gray_accum.add(1) 

        return x

    # helper functions END
    
    # set num_gray_nodes to 1 to start loop (finished when all nodes are visited)
    num_remaining_gray_nodes = 1;
    num_visited_nodes = 0;
    
    counter = 0
    gray_accum = 0
    while num_remaining_gray_nodes > 0:
    
        # (1) set accumulator for gray nodes to zero
        gray_accum = context.accumulator(0)

        # # split dataset into one of all the gray nodes 
        # # (the ones to visit next) and the remaining ones
        # rddGray = rdd.filter(lambda x: x[1][2] == 1)
        # rddRest = rdd.filter(lambda x: x[1][2] != 1)

        # # (2) start map process
        # rddGray = rddGray.flatMap(expandNode)

        # rdd = rddRest.union(rddGray)
        rdd = rdd.flatMap(expandNode)

        # (3) then reduce by key
        rdd = rdd.reduceByKey(reduceNodes)
        
        # (4) map to count gray nodes
        rdd = rdd.map(functools.partial(countGrayNodes, gray_accum=gray_accum))

        # call count to evaluate accumulator correctly
        rdd.count()
        
        # save value of gray node accumulator
        num_remaining_gray_nodes = gray_accum.value
        num_visited_nodes += num_remaining_gray_nodes 
            
        counter += 1
        
    
    # return number of visited nodes and the rdd
    return num_visited_nodes, rdd