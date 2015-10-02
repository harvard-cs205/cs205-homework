from pyspark import SparkContext

def split_link(text):
    colon_split = text.split(": ")
    return (int(colon_split[0]), map(int, colon_split[1].split(" ")))

def iterate_neighbors(node):
    vertex, neighbors = node
    min_neighbor = min(neighbors)
    new_edges = [(min_neighbor, neighbors)]
    for neighbor in neighbors:
        if neighbor != min_neighbor:
            new_edges.append((neighbor, set([min_neighbor])))
    return new_edges

def is_iterated(node):
    vertex, edges = node
    return len(edges) == 1 and vertex not in edges    

def continue_iter(node, accum):
    vertex, edges = node
    if vertex > min(edges):
        accum.add(1)

def symmetric(rdd):
    rdd = rdd.map(lambda x: (x[0], set([x[0]]) | set(x[1])))
    stop_accum = rdd.context.accumulator(1)
    while stop_accum.value > 0:
        stop_accum = rdd.context.accumulator(0)
        rdd = rdd.flatMap(iterate_neighbors).reduceByKey(lambda x,y: x|y).filter(lambda x: not is_iterated(x))
        rdd.foreach(lambda x: continue_iter(x, stop_accum))
    return {"components": rdd.count(), "max": len(rdd.takeOrdered(1, lambda x: -len(x[1]))[0][1])}

if __name__ == '__main__':
    sc = SparkContext(appName="P5")
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
    links_kv = links.map(split_link).partitionBy(64).cache()
    print "Symmetric Links:" + str(symmetric(links_kv))
    #print "Asymmetric Links:" + str(directed_links(links_kv, reduce_links))
    
