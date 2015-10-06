__author__ = 'xiaowen'
import findspark
findspark.init('/home/xiaowen/spark')
import pyspark


def get_path(links, src, snk, numPartitions):
    stp = links.context.accumulator(0)  # indicate stop
    # init path
    new_path = links.filter(lambda x: x[0] == src).map(lambda (x, y): (x, [x]))  # new path based on current frontier
    while stp.value == 0:
        # update frontier
        to_visit = links.join(new_path).partitionBy(numPartitions)  # ( index, ( [], [index1] ) )
        # update new_path
        new_path = to_visit.flatMap(
            lambda (x, (nbs, path)): [(nb, path + [nb]) for nb in nbs if nb not in path]).partitionBy(
            numPartitions)  # ( nb, [index1, nb] ) ...
        src_to_snk_n = new_path.filter(lambda (x, y): x == snk).count()
        if src_to_snk_n > 0:
            stp.add(1)
    ret = new_path.filter(lambda (x, y): x == snk).map(lambda x: x[1]).collect()
    return ret


def get_paths_names(pages, paths):
    flat_paths = [item for sublist in paths for item in sublist]
    dct = pages.filter(lambda (x, y): y in flat_paths).map(lambda (x, y): (y, x)).collectAsMap()
    return [map(lambda x: dct.get(x), path) for path in paths]


# build sparkcontext object
sc = pyspark.SparkContext(appName="P5")
# input data
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
cln_links = (links
             .map(lambda x: x.split(': '))
             .map(lambda x: (int(x[0]), x[1].split(' ')))
             .map(lambda (x, y): (x, map(int, y))))
cln_links.cache()

page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
cln_pages = page_names.zipWithIndex().map(lambda (x, y): (x, y + 1))
cln_pages.cache()
numPartitions = 64
h_id = cln_pages.lookup("Harvard_University")[0]
k_id = cln_pages.lookup("Kevin_Bacon")[0]

h_to_k = get_path(cln_links, h_id, k_id, numPartitions)
h_to_k_path_names = get_paths_names(cln_pages, h_to_k)

k_to_h = get_path(cln_links, k_id, h_id, numPartitions)
k_to_h_path_names = get_paths_names(cln_pages, k_to_h)
