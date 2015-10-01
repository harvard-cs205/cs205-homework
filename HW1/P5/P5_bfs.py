from pyspark import SparkContext

def split_link(text):
    colon_split = text.split(": ")
    return (int(colon_split[0]), map(int, colon_split[1].split(" ")))

def title_index(name1, name2, rdd):
    return (rdd.filter(lambda x: x[1] == name1).first()[0], rdd.filter(lambda x: x[1] == name2).first()[0])

def add_to_path(node, acc, index1, index2):
    new_paths = []
    last_node, pair = node
    neighbors, curr_path = pair
    for neighbor in neighbors:
        new_paths.append((neighbor, curr_path + [neighbor]))
        if index1==curr_path[0] and index2 == neighbor:
            acc.add(1)
    return new_paths

def shortest_path(index1, index2, links_rdd):
    solutions_found = links_rdd.context.accumulator(0)
    nodes_to_search = set([index1])
    path_rdd = links_rdd.filter(lambda x: x[0] in nodes_to_search).map(lambda x: (x[0], [x[0]]))
    solution_rdd = path_rdd
    while solutions_found.value == 0:
        join_rdd = links_rdd.filter(lambda x: x[0] in nodes_to_search).join(path_rdd)
        path_rdd = join_rdd.flatMap(lambda x: add_to_path(x, solutions_found, index1, index2))
        nodes_to_search = set(path_rdd.map(lambda x: x[0]).collect())
        solution_rdd = path_rdd.filter(lambda x: x[0] == index2 and x[1][0] == index1).map(lambda x: x[1]).collect()
    return solution_rdd

def get_names_path(paths, rdd):
    path_ret = []
    set_path = set([vertex for path in paths for vertex in path])
    filtered_entries = dict(rdd.filter(lambda x: x[0] in set_path).collect())
    for path in paths:
        path_ret.append(map(lambda x: filtered_entries[x], path))
    return path_ret

if __name__ == '__main__':
    sc = SparkContext(appName="P5")
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
    titles = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
    index_titles = titles.zipWithIndex().map(lambda x: (x[1] + 1, x[0])).cache()
    links_kv = links.map(split_link).cache()
    harvard_index, bacon_index = title_index('Harvard_University', 'Kevin_Bacon', index_titles)
    print get_names_path(shortest_path(harvard_index, bacon_index, links_kv), index_titles)
    print get_names_path(shortest_path(bacon_index, harvard_index, links_kv), index_titles)

