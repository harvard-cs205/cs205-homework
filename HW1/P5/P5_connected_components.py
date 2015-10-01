from pyspark import SparkContext

def split_link(text):
    colon_split = text.split(": ")
    return (int(colon_split[0]), map(int, colon_split[1].split(" ")))

def reset_key(node):
    vertices = set([node[0]]) | set(node[1])
    return (min(vertices), vertices)
def combine_key(node1, node2):
    return set(node1) | set(node2)
def create_symmetries(node):
    vertices = set([node[0]]) | set(node[1])
    paths = []
    for vertex in vertices:
        paths.append((vertex, vertices.difference(set([vertex]))))
    return paths
def reverse_edges(node):
    edges = []
    for neighbor in node[1]:
        edges.append((neighbor, set([node[0]])))
    return edges
def connected_components(rdd):
    count = rdd.count()
    while True:
        rdd = rdd.map(reset_key).reduceByKey(combine_key)
        next_count = rdd.count()
        if next_count == count:
            max_component = len(rdd.takeOrdered(1, key=lambda x: -len(x[1]))[0][1])
            return {"components": next_count, "max_component": max_component}
        count = next_count
def symmetric_links(rdd):
    #for every a->b, make sure there's a b->a
    symmetric_rdd = rdd.flatMap(create_symmetries).reduceByKey(combine_key)
    return connected_components(symmetric_rdd)
def directed_links(rdd):
    #for every a->b, remove if no b->a
    rdd_reverse = rdd.flatMap(reverse_edges).reduceByKey(combine_key)
    rdd_symmetric = rdd.join(rdd_reverse).map(lambda x: (x[0], set(x[1][0]).intersection(x[1][1])))
    return connected_components(rdd_symmetric)

if __name__ == '__main__':
    sc = SparkContext(appName="P5")
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
	
    #links_list = ["1: 2 3", "2: 1 3", "3: 1 2", "4: 1 5 6", "5: 4 6", "6: 4 5", "7: 8 9 10", "8: 7 9 10", "9: 7 8 10", "10: 7 8 9"]
    #links = sc.parallelize(range(0, 10)).map(lambda x: links_list[x])

    links_kv = links.map(split_link)
    print "Symmetric Links:" + str(symmetric_links(links_kv))
    print "Asymmetric Links:" + str(directed_links(links_kv))
	


