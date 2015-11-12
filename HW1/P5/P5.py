# Possible to use it locally on ipython with P5.ipynb
# Parameters
partitions = 64

# Graph of the links to: (page_id, [to_id])
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
link_to_graph = links.map(lambda line: (int(line.split(':')[0]),
                          [int(v) for v in line.split(':')[1].split(' ')[1:]]))\
    .partitionBy(partitions).cache()

# Graph of the names: (page_name, page_id)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
page_names = page_names.zipWithIndex().mapValues(lambda v: v + 1)\
    .partitionBy(partitions).cache()

# Creating the reverse graph, ie the links from
link_from_graph = link_to_graph.flatMap(lambda x: [(to, x[0]) for to in x[1]])\
    .groupByKey().map(lambda x: (x[0], set(list(x[1])))).cache()


# ## Part A

def get_path(x):
    value = list(x[1])
    length = [len(n) for n in value]
    ind = min(length)
    path = value[length.index(ind)]
    return (x[0], path)


def check_target(x, acc):
    if x[0] == target:
        acc.add(1)


def shortest_path(graph, root, target, partitions=64):
    '''
    Return the shortest_path as a list between the root and the target of the
    graph if the two are connected.
    Otherwise it returns an empty list.
    '''
    # Initialization
    i = 0
    target_found = graph.context.accumulator(0)
    response = []

    # Graph used to store the next nodes to visit.
    # Format is (name, (path_from_root, neighbors))
    next_nodes = graph.filter(lambda x: x[0] == root)\
        .mapValues(lambda v: ([], v))
    # Graph used to store the path from the root for all visited nodes.
    # Format is (name, path_from_root)
    path = graph.context.parallelize([(root, [])]).partitionBy(partitions)
    while target_found.value == 0:
        print("Loop number {}".format(i))
        # Visiting the neighbors of the Nodes in the subgraph current and
        # updating their distance.
        visiting = next_nodes\
            .flatMap(lambda x: [(n, x[1][0] + [x[0]]) for n in list(x[1][1])])\
            .partitionBy(partitions)
        path = visiting.union(path)\
            .groupByKey().map(get_path).partitionBy(partitions)
        i += 1
        # Getting only the current visited nodes (not previously visited)
        visiting_ = path.filter(lambda x: len(x[1]) == i)
        # Check if target hit
        visiting_.foreach(lambda x: check_target(x, target_found))
        # Getting the list of the neighbors of the current nodes as values.
        next_nodes = visiting_.join(graph)
        # Correct path if target has been found
        response = path.filter(lambda x: x[0] == target)
    return response


# Printing the paths with names:
def get_name(node_id):
    return page_names.filter(lambda x: x[1] == node_id).keys().take(1)[0]


def get_path_names(path):
    path_list = []
    node, parents = path.collect()[0]
    for p in parents:
        path_list.append(get_name(p))
    path_list.append(get_name(node))
    return path_list


# ## Example
root_name = 'Kevin_Bacon'
target_name = 'Harvard_University'

root = page_names.lookup(root_name)[0]
target = page_names.lookup(target_name)[0]

path = shortest_path(link_to_graph, root, target)

path_names = get_path_names(path)
