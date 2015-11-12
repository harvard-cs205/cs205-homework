from pyspark import SparkContext, SparkConf
sc = SparkContext()

# Path between two nodes (assumes one exists!)
def path_finder(start_name, end_name, page_links, page_names, sc):
    start_id = page_names.filter(lambda x: x[1] == start_name).take(1)[0][0]
    end_id = page_names.filter(lambda x: x[1] == end_name).take(1)[0][0]
    
    paths = sc.parallelize([])
    frontier_links = sc.parallelize([start_id])
    
    itr_acc = sc.accumulator(0)
    while True:
        itr_acc+=1
        print "Entering iteration: ", itr_acc.value
        # Get new neighbor tuples (remove dupes with same end value)
        new_paths = frontier_links.map(lambda x: (x, [])).join(page_links)
        new_paths_flat = new_paths.flatMapValues(lambda node: node[1]).map(lambda x: (x[1], x[0]))
        new_paths_unique_r = new_paths_flat.reduceByKey(lambda x,y: x)
        paths_r = paths.map(lambda x: (x[1], x[0]))
        
        # Remove path tuples whose value is already the key of 
        # a tuple in our paths list (this would just be a loop, no new info)
        new_paths_unique_r = new_paths_unique_r.subtractByKey(paths)
        new_paths_unique_r = new_paths_unique_r.subtractByKey(paths_r)
            
        new_paths_unique = new_paths_unique_r.map(lambda x: (x[1], x[0]))
        paths = paths.union(new_paths_unique).partitionBy(32).cache()
        frontier_links = new_paths_unique.values()
        
        # Short-circuit if the end_id is in the frontier list
        if (not (frontier_links.filter(lambda x: x == end_id).isEmpty())):
            print "End ID in frontier, breaking now!"
            break
        print "Not in frontier...running another iteration"
        
    # Work backwords through the graph from the end value to the start
    path_list = []
    last_node = end_id
    path_list.append(last_node)
    
    while True:
        last_node = paths.filter(lambda x: x[1] == last_node).take(1)
        last_node = last_node[0][0]
        path_list.append(last_node)
        if last_node == start_id:
            break
    
    # Reverse the list
    path_list.reverse()

    # Now get the names for the ids to build the actual path!
    path_list_names = []
    for id in path_list:
        path_list_names.append(page_names.lookup(id)[0])
        
    print "Here is a shortest path between %s and %s: " % (start_name, end_name)
    print path_list_names
    return path_list_names

###################
## SHORTEST PATH ##
###################

# Make page_names
# titles-sorted.txt
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt').zipWithIndex().map(lambda x: (x[1] + 1, x[0])).partitionBy(32).cache()

# Make page_links and flatten
# links-simple-sorted.txt
# from1: to11 to12 to13
page_links_with_strings = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt').map(lambda line: (int(line[:line.rindex(':')]), line[line.rindex(':') + 2:]))
# (1, 1664968), (2, 3), (2, 747213), (2, 1664968), (2, 1691047)
page_links = page_links_with_strings.map(lambda line: (line[0], [int(x) for x in line[1].split(' ')])).partitionBy(32).cache()

# Run some SSBFS searches
kevinToHarvard = path_finder('Kevin_Bacon','Harvard_University', page_links, page_names, sc)
harvardToKevin = path_finder('Harvard_University','Kevin_Bacon', page_links, page_names, sc)

# Just for fun
margaretToLizardPeople = path_finder('Margaret_Thatcher','Lizardfolk', page_links, page_names, sc)