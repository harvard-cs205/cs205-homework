from pyspark import SparkContext

def parse_text(line):
    data = line.split(': ')
    return (int(data[0]), set(map(int, data[1].split())))

def index_clusters((parent, children)):
    # use min of neighbors to index a cluster
    return min(children), children

def rep_clusters((parent, children)):
    children.add(parent)
    return children, children

def union(x, y):
    return x | y

def iter_clusters(clusters):
    # find connected components by grouping all connected nodes together
    # indexed by min node
    return clusters.map(rep_clusters).flatMapValues(lambda x: x).map(lambda (k,v): (v,k)).reduceByKey(union).map(index_clusters).reduceByKey(union)

def onnected_components(clusters):
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

    # convert to symmetric graph
    symmetric_graph = links.flatMapValues(lambda x: x).map(lambda (parent, child): (child, set([parent]))).union(links).reduceByKey(lambda x, y: x|y).cache()
    num, max_size = onnected_components(symmetric_graph)
    print 'No. of connected components:', num
    print 'No. of nodes in the largest connected component:', max_size