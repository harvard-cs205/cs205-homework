#import findspark
#findspark.init()
#import pyspark
from pyspark import SparkContext, SparkConf
#conf = SparkConf().setAppName("app").setMaster("local")
sc = SparkContext(appName='test')
def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)
#quiet_logs(sc)
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',40)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt',40)
import time
start = time.time()
def links_mapper(w):
    if ': ' not in w:
        return -1, -1
    w = w.split(': ')
    index = int(w[0])
    neighbors = []
    for item in w[1].split(' '):
        if len(item) > 0:
            neighbors.append(int(item))
    return index, neighbors

page_index_RDD = page_names.zipWithIndex().map(lambda (k, v): (k, v+1))
page_index_RDD.cache()
links_RDD = links.map(links_mapper)
links_RDD.cache()


def bfs(origin, target):
    global sc, page_index_RDD, links_RDD
    origin_index = page_index_RDD.lookup(origin)[0]
    target_index = page_index_RDD.lookup(target)[0]
    max_distance = 10000000

    # construct graphRDD
    def construct_mapper((key, neighbors)):
        if key == origin_index:
            return key, (True, neighbors, 0, [])
        return key, (False, neighbors, max_distance, [])

    graph_rdd = links_RDD.map(construct_mapper).cache()
    present_rdd = graph_rdd

    def mapper((key, (flag, neighbors, distance, pre_nodes))):
        result = list()
        result.append((key, (False, neighbors, distance, pre_nodes)))
        if distance == turn_id:
            tmp = [x for x in pre_nodes]
            tmp.append(key)
            for otherKey in neighbors:
                result.append((otherKey, (True, [], distance+1, tmp)))
        return result

    def reducer((flag1, adj1, dist1, pre1), (flag2, adj2, dist2, pre2)):
        adj = adj1 if len(adj1) > len(adj2) else adj2
        if dist1 < dist2:
            return flag1, adj, dist1, pre1
        else:
            return flag2, adj, dist2, pre2

    turn_id, is_found = 0, []
    while True:
        last_num = present_rdd.filter(lambda (k, (f, a, d, p)): d == turn_id + 1).count()
        present_rdd = present_rdd.flatMap(mapper).reduceByKey(reducer)
        is_found = present_rdd.filter(lambda (k, (f, a, d, p)): k == target_index and d != max_distance).collect()
        if len(is_found) > 0:
            break
        present_num = present_rdd.filter(lambda (k, (f, a, d, p)): d == turn_id + 1).count()
        if present_num == last_num:
            break
        else:
            turn_id += 1
    return present_rdd, is_found

o = 'Harvard_University'
t = 'Kevin_Bacon'
final_rdd_o2t, res_o2t = bfs(o, t)
final_rdd_t2o, res_t2o = bfs(t, o)
path_o2t = []
path_t2o = []
for i in res_o2t[0][1][3]:
    path_o2t.append(page_index_RDD.map(lambda (k, v): (v,k)).lookup(i))

for j in res_t2o[0][1][3]:
    path_t2o.append(page_index_RDD.map(lambda (k, v): (v,k)).lookup(j))


log_file = open('P5_bfs_res_final.txt', 'w')
log_file.write('Harvard 2 Bacon:\n' + str(res_o2t) + '\n\n\n')
log_file.write('Bacon 2 Harvard:\n' + str(res_t2o) + '\n\n\n')
for i in path_t2o:
    log_file.write(str(i))
    log_file.write('--------->')
log_file.write(o+'(arrived!)'+'\n\n\n\n\n\n')
for i in path_o2t:
    log_file.write(str(i))
    log_file.write('--------->')
log_file.write(t+'(arrived!)'+'\n\n\n\n\n\n')
print time.time()-start
log_file.write('running time is'+str(time.time()-start)+'\n\n')
log_file.close()
