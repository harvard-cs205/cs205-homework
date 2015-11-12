import pyspark

def Graph_BFS(graph, v_str, limitedDiam,sc):
    
    # function for flatMaping certain level of nodes to a series of nodes
    def DistrFunc(node):
        dist=node[1][1]
        currNode=node[0]
        listNeighbors=node[1][0]
        result=[node]
        if dist==currLevel:
            for neighbor in listNeighbors:
                result.append((neighbor, ([],dist+1)))
        return result
    
    # function for reduceByKey, updating the the distance and neighbor list
    def reduceFunc(var1,var2):
        dist1=var1[1]
        neighbor1=var1[0]
    
        dist2=var2[1]
        neighbor2=var2[0]
    
        if dist1<dist2:
            d=dist1
        else:
            d=dist2
    
        neighbor=neighbor1+neighbor2
        return (neighbor, d)
    
    # BFS start:
    # preprocessing, map each node in graph rdd to tuple
    SG=graph.map(lambda node: (node[0],(node[1], 0)) if node[0]==v_str else (node[0],(node[1], 1000)))
    Diameter=10
    currLevel=0  # current level
    numVisitedNodes = sc.accumulator(1)
    if limitedDiam:  # is diameter limited? 
        while currLevel<Diameter:
            SG=SG.flatMap(DistrFunc)
            SG=SG.reduceByKey(reduceFunc)
            NewlyVisited=SG.filter(lambda node: node[1][1]==currLevel+1).count()
            print 'Current Level: ', currLevel, 'Newly Visited Nodes: ', NewlyVisited, '\n'
            numVisitedNodes.add(NewlyVisited)
            currLevel=currLevel+1
    else:
        while True:
            SG=SG.flatMap(DistrFunc)
            SG=SG.reduceByKey(reduceFunc)
            NewlyVisited=SG.filter(lambda node: node[1][1]==currLevel+1).count()
            print 'Current Level: ', currLevel, 'Newly Visited Nodes: ', NewlyVisited, '\n'
            numVisitedNodes.add(NewlyVisited)
            if NewlyVisited==0:
                break
            else:
                currLevel=currLevel+1

    print 'Total Nodes Touched: ', numVisitedNodes.value
