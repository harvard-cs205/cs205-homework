# Author: George Lok
# P5_bfs.py

import time

def createFlatMapFunctor(visited) :

    # Semantically, increments the frontier,
    # ignoring nodes we've already visited.
    def flatMapFunctor((source, (paths, dests))) :
        results = []
        for dest in dests :
            if dest in visited :
                continue
            for path in paths :
                newJump = (dest,path + [source])   
                results.append(newJump)
        return results
    return flatMapFunctor
                       
def filter ((source, paths)) :
     return paths is not None

# Source Destination Breadth First Search
def SDBFS(source, dest, graph, sc) :
    # Using 2 executors with 4 cores, so 32 partitions seems reasonable. 
    # Paths will store our shortest paths if they exist.  
    paths = graph.keys().map(lambda x : (x,[[]])).partitionBy(32)
    
    # In order to allow this to find self loops (i.e. source == dest), we prepare the frontier
    # to be the nodes adjacent to source (i.e. we do not visit source)
    firstJumps = []
    firstBranch = graph.lookup(source)[0]
    for item in firstBranch :
        firstJumps.append((item, [[source]]))
    
    # nextJumps is our frontier, represented with an RDD
    nextJumps = sc.parallelize(firstJumps)

    count = 0
    visited = set()
    while True :
        print "Starting Iteration: " + str(count)

        # Add paths stored in our frontier to our paths RDD
        paths = paths.leftOuterJoin(nextJumps).map(lambda (x,(y, z)) : (x, z)).partitionBy(32)

        # Mark paths that have just been updated
        print "Begin Filtering"
        filteredPaths = paths.filter(filter)

        # Perform our computations
        filteredKeysList = filteredPaths.keys().collect()

        # Check if we should end
        if len(filteredKeysList) == 0 :
            return []

        # Update visited nodes, and see whether the frontier found our destination.
        for key in filteredKeysList :
            if key == dest :
                allPaths = filteredPaths.lookup(key)[0]
                for path in allPaths :
                    path.append(key)
                return allPaths
            visited.add(key)
        
        # We need to do another iteration, so join with the adj list to get the next nodes for the frontier
        filteredPathsJoined = filteredPaths.join(graph)

        print "Starting Flatmap"
        start = time.time()

        # Generate frontier RDD
        nextJumps = filteredPathsJoined.flatMap(createFlatMapFunctor(set(visited))).groupByKey().map(lambda (x,y) : (x, list(y)))
        
        end = time.time()
        print "flatMap Time " + str(end - start) 
        count += 1