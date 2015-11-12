def bfs(sourceRDD, root, step_limit = False):
    ## Set the root
    def set_root((node, (dist, neighbors))):
        if (node == root):
            return (node, (0, neighbors))
        else:
            return ((node, (dist, neighbors)))

    ## Update the node RDD: character, (distance, [list of neighbors])
    ## First we find the current node's neighbors and create a row: neighbor, (current_distance + 1, [empty list])
    ## Then we combine the newly created neighbor with the old neighbor with the updated distance
        ## min(dist1,dist2) ensures the reached nodes keep the original distance
        ## y1+y2 keeps the value of the neighbor list, concatinate the old neighbor list with the empty list created at this step
    def update_node(inputRDD, step, visit_count):
        def find_neighbor((node, (dist, neighbors))):
            next_list = [(node, (dist, neighbors))]
            if dist == step:
                for neighbor in neighbors:
                    next_list.append((neighbor, (step+1, [])))
            return next_list
        outputRDD = inputRDD.flatMap(find_neighbor).reduceByKey(lambda (x1,y1), (x2,y2): (min(x1,x2), y1+y2))
        return outputRDD    

    ## Initialize root and count variables
    nodeRDD = sourceRDD.map((set_root))
    new_count = 1
    visit_count = sc.accumulator(1)
    print root

    ## Depending on if the number of searches is limited, call the bfs multiple times and update the count variables
    if step_limit != False:
        for step in range(step_limit):
            nodeRDD = update_node(nodeRDD, step, visit_count)
            new_count = nodeRDD.filter(lambda (k, v): v[0] == step+1).count()
            visit_count += new_count        
            print "Step: ", (step+1), "; New Nodes: ", new_count, "; Total Nodes: ", visit_count, "."
        return (step+1, visit_count, nodeRDD)
    else:
        step = 0
        while new_count > 0:
            nodeRDD = update_node(nodeRDD, step, visit_count)
            new_count = nodeRDD.filter(lambda (k, v): v[0] == step+1).count()
            visit_count += new_count
            print "Step: ", (step+1), "; New Nodes: ", new_count, "; Total Nodes: ", visit_count, "."
            if new_count == 0:
                return (step+1, visit_count, nodeRDD)