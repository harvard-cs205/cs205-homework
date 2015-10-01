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
    path_rdd = links_rdd.map(lambda x: (x[0], [x[0]]))
    solution_rdd = path_rdd
    while solutions_found.value == 0:
        join_rdd = links_rdd.join(path_rdd)
        path_rdd = join_rdd.flatMap(lambda x: add_to_path(x, solutions_found, index1, index2))
        solution_rdd = path_rdd.filter(lambda x: x[0] == index2 and x[1][0] == index1).map(lambda x: x[1]).collect()
    return solution_rdd

def get_names_path(paths, rdd):
    path_ret = []
    for path in paths:
        set_path = set(path)
        filtered_entries = dict(rdd.filter(lambda x: x[0] in set_path).collect())
        path_ret.append(map(lambda x: filtered_entries[x], path))
    return path_ret

if __name__ == '__main__':
	sc = SparkContext(appName="P5")
	#links_list = ['2: 5 7 4', '1: 5 1 7', '10: 6 9 7', '4: 8 10 4', '7: 3 7 4', '3: 3 7 2', '6: 6 2 4', '5: 3 2 1', '1: 2 6 9', '10: 9 1 7']
	#titles_list = ['AKG', 'MOB', 'INJ', 'HLB', 'DCE', 'COL', 'GKI', 'GKN', 'FJI', 'JKA']
	links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
	titles = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
	#links = sc.parallelize(range(0, 10)).map(lambda x: links_list[x])
	#titles = sc.parallelize(range(0, 10)).map(lambda x: titles_list[x])
	index_titles = titles.zipWithIndex().map(lambda x: (x[1] + 1, x[0]))
	links_kv = links.map(split_link)
	harvard_index, bacon_index = title_index('Harvard_University', 'Kevin_Bacon', index_titles)
	#harvard_index, bacon_index = title_index(titles_list[1], titles_list[5], index_titles)
	print get_names_path(shortest_path(harvard_index, bacon_index, links_kv), index_titles)


