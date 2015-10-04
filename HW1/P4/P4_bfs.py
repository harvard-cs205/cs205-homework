import findspark
findspark.init()
import pyspark
import json

def bfs_core(presentRDD, numOfVisitedNodes, isDiameterLimited = True):
    
    # In turn i+1, we update neighbors of nodes with distance i,
    # and find all nodes whose distance is i+1 from origin

    # graphRDD's item (key, (flag, AdjacentNodes, distance)) flag means whether this node is active 
    # being active means this key-value pair will be used for updating some nodes' distance in this turn
    # for each item in graphRDD, we emit two types of (key, value) pair
    # first type simply consists of itself (for we don't want to lose the adjacentNodes information during each turn)
    # second type will be send only by those nodes that are new visited nodes in the previous turn. We are going to update distances of these nodes' unvisited neighbors

    def mapFunc((key, (flag, AdjacentNodes, distance))):
        result = []
        if distance == turnId:
            for otherKey in AdjacentNodes:
                result.append((otherKey, (True, [], distance+1)))
        result.append((key, (False, AdjacentNodes, distance)))
        return result

    # After map, for each key (actor's unique index), we may have one or more than one key-value pairs
    # For one key (exactly one node or one actor)
    # If there's only one key-value pair, then by mapFunc we know this turn has nothing to do with this key
    # so we simply return it without any modification
    # If there's more than one key-value pair, if one of them has a True flag, we know this key's distance should be updated
    # so we return the smallest distance in all distances of corresponding values

    def reduceFunc((flag1, Adj1, dist1), (flag2, Adj2, dist2)):
        if flag1 is True and flag2 is False:
            return (True, Adj2, dist1 if dist1 < dist2 else dist2)
        elif flag2 is True and flag1 is False:
            return (True, Adj1, dist1 if dist1 < dist2 else dist2)
        elif flag1 is True and flag2 is True:
            return (True, Adj1 if len(Adj1) > len(Adj2) else Adj2 , dist1 if dist1 < dist2 else dist2)
        #if flag1 or flag2:
            #return (True, Adj1, dist1 if dist1 < dist2 else dist2)
        else:
            return (False, Adj1 if len(Adj1) > len(Adj2) else Adj2 , dist1) #obj1)

    # After turn i+1's reduce, we update(/find) all nodes whose distance is (i+1) from origin

    # When we assume diameter is limited, there's nothing special and we just update whole graph turn by turn
    # When assumption is relaxed, if we update distance of 0 new node in turn i, we know we should stop at here
    # We use numOfVisitedNodes (type sc.accumulator) to represent all nodes whose distances have been updated in present turn
    # We use presentRDD.filter(lambda (k, (f, a, d)): d == turnId + 1).count() to get the num of new visited nodes in turn [turnId]
    # here (k, (f, a, d)) is key-value pair of RDD in each turn, which is the short version of (key, (flag, adjacentNodes, distance))
    log = ''
    if isDiameterLimited:
        for turnId in range(0, 10):
            #print presentRDD.count()
            presentRDD = presentRDD.flatMap(mapFunc).reduceByKey(reduceFunc)
            numOfNewVisitedNodes = presentRDD.filter(lambda (k, (f, a, d)): d == turnId + 1).count()
            numOfVisitedNodes.add(numOfNewVisitedNodes)
            log += 'After ' + str(turnId + 1) + ' turn, the number of new visited nodes is ' + str(numOfNewVisitedNodes) + '\n'
    else:
        turnId = 0
        while True:
            presentRDD = presentRDD.flatMap(mapFunc).reduceByKey(reduceFunc)
            numOfNewVisitedNodes = presentRDD.filter(lambda (k, (f, a, d)): d == turnId + 1).count()
            numOfVisitedNodes.add(numOfNewVisitedNodes)
            #print numOfVisitedNodes
            log += 'After ' + str(turnId + 1) + ' turn, the number of new visited nodes is ' + str(numOfNewVisitedNodes) + '\n'
            if numOfNewVisitedNodes == 0:
                #print presentRDD.filter(lambda x:x[1][2]==1000).count()
                break
            else:
                turnId += 1
    # return value is a tuple (total num of turns, num of all visited nodes, log) 
    # log is a string contains how many new visited nodes of each turn
    return (10 if isDiameterLimited else turnId, numOfVisitedNodes.value, log)

# graphRDD's item should be [A, B1, B2, ...] where nodes B1, B2, ... are all connected to node A
# final function, originActor is the root actor's name (type str)
# isDiameterLimited represents whether diameter assumption is relaxed
def bfs(sc,graphRDD, origin, isDiameterLimited = False):
    # W.O.L.G, for those nodes unvisited, we set their distance to maxDist
    maxDist = 1000
    def preProcessMap(nodeList):
        if (nodeList[0] == origin):
            return (nodeList[0], (True, nodeList[1:], 0))
        else:
            return (nodeList[0], (False, nodeList[1:], maxDist))
    graphRDD = graphRDD.map(preProcessMap)

    numOfVisitedNodes = sc.accumulator(1) # construct the accumulator, represents the number of visited nodes
    return bfs_core(graphRDD, numOfVisitedNodes, isDiameterLimited)
