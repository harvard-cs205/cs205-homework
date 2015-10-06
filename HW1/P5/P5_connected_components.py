__author__ = 'xiaowen'

from sys import maxint
import numpy as np
import findspark
findspark.init('/home/xiaowen/spark')
import pyspark


def add_reverse(x):
    node = x[0]
    nbs = x[1]
    return [(nb, set([node])) for nb in nbs] + [(node, set(nbs))]


def reverse(x):
    node = x[0]
    nbs = x[1]
    return [(nb, set([node])) for nb in nbs]


def get_bfs(neighbors, source):
    bfs = (neighbors
           .map(lambda (x, y): (x, (maxint, list(y), None)) if x != source else (x, (0, list(y), []))))
    return bfs


def get_table(bfs):
    tab = None
    try:
        tab = bfs.map(lambda (x, y): (x, (y[0], y[2]))).collectAsMap()
    except Exception:
        print bfs.take(4)
    return tab


def calc_bfs(bfs, tabs):
    ave = 0.0
    stp = sc.accumulator(0)
    while stp.value == 0:
        # print(tabs)
        # filter, only the frontier is left
        # update frontier's neighbors
        bfs = bfs.map(lambda x: updateDistance(x, tabs))
        tabs = get_table(bfs)
        prev = ave
        ave = np.mean([it[0] for it in tabs.values()])
        if prev == ave or stp:
            # print(distances)
            break
    return bfs


def get_dis_path(x, tabs):
    tup = tabs.get(x)
    if tup == None or tup[0] == maxint:
        return maxint, None
    # print(tup)
    # print(x)
    ret = (tup[0] + 1, tup[1] + [x])
    return ret


def updateDistance(x, tabs):
    node = x[0]
    dis = x[1][0]
    neighbors = x[1][1]
    paths = x[1][2]
    if dis != maxint or len(neighbors) == 0:
        return node, (dis, neighbors, paths)
    # distances of neighbors
    dp_lst = [get_dis_path(nb, tabs) for nb in neighbors]
    # update my distance with min + 1 if it is smaller
    mn = dp_lst[0]
    for dp in dp_lst:
        if mn[0] > dp[0]:
            mn = dp
    # driverDistances.add((node, dis))
    return node, (mn[0], neighbors, mn[1])


def get_components(neighbors):
    total = set(neighbors.map(lambda (x, y): x).collect())
    cnt = []
    while len(total) > 0:
        node = total.pop()
        bfs = get_bfs(neighbors, node)
        tabs = get_table(bfs)
        r = calc_bfs(bfs, tabs)
        cur = set(r.filter(lambda (node, (d, lst, path)): d < maxint).map(lambda x: x[0]).collect())
        cnt.append(len(cur))
        # print(cnt)
        total = total - cur
        # print len(total)
        total_copy = total.copy()
        neighbors = neighbors.filter(lambda (x, nb): x in total_copy)
    return cnt


def mksym_wrapper(links):
    mksym_links = links.flatMap(add_reverse).reduceByKey(lambda x, y: x.union(y))
    mksym_comps = get_components(mksym_links)
    return mksym_comps


def sym_wrapper(links):
    rdd_reverse = links.flatMap(reverse).reduceByKey(lambda x, y: x.union(y))
    rdd_symmetric = links.map(lambda (x, y): (x, set(y))).join(rdd_reverse).map(
        lambda (x, (a, b)): (x, a.intersection(b)))
    return get_components(rdd_symmetric)


# build sparkcontext object
sc = pyspark.SparkContext(appName="P5")
# input data
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
cln_links = (links
             .map(lambda x: x.split(': '))
             .map(lambda x: (int(x[0]), x[1].split(' '))))

cln_links = cln_links.cache()

sym_comps = sym_wrapper(cln_links)
mksym_comps = mksym_wrapper(cln_links)


print 'Symmetric:', 'number:', len(sym_comps), 'max:', max(sym_comps)
print 'Asymmetric:', 'number:', len(mksym_comps), 'max:', max(mksym_comps)


