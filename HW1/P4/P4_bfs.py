def edgeMap(x) :
    return (x[0], set(x[1]), -1)

def createMapFunctor(count, nextNodes) :
    def mapFunctor((k, s, c,)) :
        if k in nextNodes :
            # We have the constraint that count is never
            if c < 0 :
                return (k, s,count)
        return (k, s,c)
    return mapFunctor

def SSBFS(source, graph, sc) :
    ac = sc.accumulator(0)
    
    def reduceFunctor((k1, s1, c1), (k2, s2, c2)) :
        ac.add(1)
        return (0, s1 | s2, 0 )

    nodes = graph.map(edgeMap)
    nextNodes = set([source])
    count = 0
    last = 0
    while len(nextNodes) > 0:
        last = ac.value
        nodes = nodes.map(createMapFunctor(count, nextNodes))
        def filterFunctor(x) :
            return x[2] == count
        changedNodes = nodes.filter(filterFunctor)
    
        if changedNodes.count() == 0 :
            break
        ac.add(1)
        reduceResult = changedNodes.reduce(reduceFunctor)
        nextNodes = reduceResult[1]
        count += 1
    return ac.value, nodes
