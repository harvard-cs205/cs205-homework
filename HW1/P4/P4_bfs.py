def p4bfs(start, rdd):
    # Create rdd with all characters
    character = start.map(lambda x: (x[0], 1))
    char_collect = character.collect()
    # find node 0 
    node_0 = start.filter(lambda x: x[0] == rdd)  
    # subtract from character rdd
    character = character.subtractByKey(node_0)
    # Find the nodes at a distance 1 
    node_1 = node_0.flatMapValues(lambda x: x).map(lambda g: (g[1], 1))
    # update character rdd by subtracting the characters just found
    character = character.subtractByKey(node_1)
    character_old = len(character.collect())
    # At this point we look for nodes at a distance 2
    i = 2
    # now we gonna iterate until d = 10
    while i < 10:
         # Find the nodes at a distance d
         node_1 = start.join(node_1).map(lambda x: (x[0], x[1][0])).flatMapValues(lambda x: 
            x).map(lambda g: (g[1], 1)).groupByKey()
         node_1 = node_1.map(lambda x: (x[0], 1))       
         character = character.subtractByKey(node_1)
         character_new = len(character.collect())       
         if character_new < character_old:
            i = i + 1
            character_old = character_new
         else:
            break
    return len(char_collect), character_new, i - 1