__author__ = 'xiaowen'

import numpy as np
import findspark
findspark.init('/home/xiaowen/spark')
import pyspark
from sys import maxint


def get_graph(rawData):
    heroToComic = rawData.map(lambda x: [it.replace('"', '') for it in x.split('","')])
    comicToHero = heroToComic.map(lambda x: (x[1], [x[0].strip()]))
    # comicToHero = heroToComic.map(lambda x: (x[1], [x[0]]))
    comicToHeros = comicToHero.reduceByKey(lambda x, y: x + y)
    neighbors = (comicToHeros
                 .flatMap(lambda (x, y): [(it, set(y)) for it in y])
                 .reduceByKey(lambda x, y: x.union(y)))
    return neighbors


def get_bfs(neighbors, source):
    bfs = (neighbors
           .map(lambda (x, y): (x, (maxint, list(y))) if x != source else (x, (0, list(y)))))
    return bfs


def get_dis(bfs):
    distances = bfs.map(lambda (x, y): (x, y[0])).collectAsMap()
    return distances


def updateDistance(node, dis, neighbors, distances):
    # distances of neighbors
    nDist = map(lambda x: distances[x], neighbors)
    # get the min
    minD = min(nDist) + 1
    # update my distance with min + 1 if it is smaller
    if minD < dis:
        dis = minD
    return node, (dis, neighbors)


def calc_bfs(bfs, distances):
    ave = 0.0
    while True:
        bfs = bfs.map(lambda (node, (dis, neighbors)): updateDistance(node, dis, neighbors, distances) if dis == maxint else (node, (dis, neighbors)))
        distances = bfs.map(lambda (node, (dis, nghb)): (node, dis)).collectAsMap()
        prev = ave
        ave = np.mean(distances.values())
        if prev == ave:
            break
    return bfs


def get_touched(neighbors, source):
    bfs = get_bfs(neighbors, source)
    distances = get_dis(bfs)
    bfs_r = calc_bfs(bfs, distances)
    return bfs_r.filter(lambda (node, (d, lst)): d < maxint).map(lambda (x, y): x).collect()


def get_comp(neighbors):
    total = set(neighbors.map(lambda (x, y): x).collect())
    print('total=%d' % len(total))
    comp_sz = []
    while len(total) != 0:
        neighbors = neighbors.filter(lambda (x, y): x in total)

        seed = list(total)[0]
        # print(sum(comp_sz))
        try:
            comp = set(get_touched(neighbors, seed))
        except:
            neighbors = neighbors.repartition(numPartitions)
            comp = set(get_touched(neighbors, seed))
        comp_sz.append(len(comp))
        total = total - comp
    return comp_sz


def add_reverse(x):
    node = x[0]
    nbs = x[1]
    return [(nb, set([node])) for nb in nbs] + [(node, set(nbs))]


def reverse(x):
    node = x[0]
    nbs = x[1]
    return [(nb, set([node])) for nb in nbs]


def mksym_wrapper(links):
    mksym_links = links.flatMap(add_reverse).reduceByKey(lambda x, y: x.union(y))
    mksym_comps = get_comp(mksym_links)
    return mksym_comps


def sym_wrapper(links):
    rdd_reverse = links.flatMap(reverse).reduceByKey(lambda x,y: x.union(y)).partitionBy(numPartitions)
    links = links.partitionBy(numPartitions)
    rdd_symmetric = links.map(lambda (x, y): (x, set(y))).join(rdd_reverse).map(lambda (x, (a, b)): (x, a.intersection(b)))
    rdd_symmetric = rdd_symmetric.filter(lambda (x, y): len(y) > 0)
    return get_comp(rdd_symmetric)


sc = pyspark.SparkContext(appName="P5")
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
cln_links = (links
             .map(lambda x: x.split(': '))
             .map(lambda x: (int(x[0]), x[1].split(' ')))
             .map(lambda (x, y): (x, map(int, y))))
cln_links = cln_links.map(lambda (x, y): (x, set(y)))
cln_links.cache()
numPartitions = 8
mksym_comps = mksym_wrapper(cln_links)
sym_comps = sym_wrapper(cln_links)
mksym_comps
sym_comps


print 'Symmetric:', 'number:', len(sym_comps), 'max:', max(sym_comps)
print 'Asymmetric:', 'number:', len(mksym_comps), 'max:', max(mksym_comps)


