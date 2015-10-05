from pyspark import SparkContext

def parse_text(line):
    data = line.split(': ')
    return (int(data[0]), set(map(int, data[1].split())))

def index_clusters((parent, children)):
    # use min of neighbors to index a cluster
    return min(children), children

def rep_clusters((parent, children)):
    return children, children

def union(x, y):
    return x | y

def I(x):
    return x

def swapkv((k, v)):
    return v, k

def iter_clusters(clusters):
    # find connected components by grouping all connected nodes together
    # each connected component is indexed by min node
    return clusters.map(rep_clusters).flatMapValues(I).map(swapkv).reduceByKey(union).map(index_clusters).reduceByKey(union)

def connected_components(clusters):
    # iterate through all clusters until convergence
    while True:
        prior = clusters.count()
        clusters = iter_clusters(clusters)
        curr = clusters.count()
        if prior == curr:
            break
    # largest size of connected components
    max_size = len(clusters.max(key=lambda (idx, cluster): len(cluster))[1])
    return curr, max_size

if __name__ == '__main__':
    # set up partitioning params
    NPART = 9*16

    # initialize spark
    sc = SparkContext("local", appName="CC")

    # load files
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', NPART).map(parse_text)
    # links = sc.textFile('links-simple-sorted.txt', NPART).map(parse_text)

    forward_links = links.flatMapValues(I).map(swapkv)
    backward_links = links.flatMapValues(I).map(swapkv)
    
    # convert to symmetric links
    symmetric_links = forward_links.union(backward_links).groupByKey().map(lambda (parent, child): (parent, set(child))).cache()
    # print symmetric_links.collect()

    # convert to bidirectional links
    bidirectional_links = forward_links.intersection(backward_links).groupByKey().map(lambda (parent, child): (parent, set(child))).cache()
    # print bidirectional_links.collect()
    
    # run connected components for symmetric links
    num, max_size = connected_components(symmetric_links)
    print 'Symmetric links'
    print 'No. of connected components:', num
    print 'No. of nodes in the largest connected component:', max_size

    # run connected components for bidirectional links
    num, max_size = connected_components(bidirectional_links)
    print 'Bidirectional links'
    print 'No. of connected components:', num
    print 'No. of nodes in the largest connected component:', max_size