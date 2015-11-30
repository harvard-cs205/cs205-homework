def BFS(sc,rdd, node, n):
    out_rdd = sc.parallelize([(node,0)])
    for i in range(0,n+1):
        n_RDD = out_rdd.filter(lambda x:x[1] == i).join(rdd).flatMap(lambda x: x[1][1]).distinct()
        #print out_rdd.count()
        if n_RDD.isEmpty():
            break
        n_RDD = n_RDD.map(lambda x:(x,i+1))
        out_rdd = out_rdd.union(n_RDD)
        out_rdd = out_rdd.reduceByKey(lambda x,y:min(x,y))
    return out_rdd

def BFS_acc(sc,rdd, node):
    out_rdd = sc.parallelize([(node,0)])
    it_count = sc.accumulator(0)
    while True:
        dist = it_count.value
        n_RDD = out_rdd.filter(lambda x:x[1] == dist).join(rdd)\
                .flatMap(lambda x: x[1][1]).distinct()
        #print out_rdd.count()
        if n_RDD.isEmpty():
            break
        n_RDD = n_RDD.map(lambda x:(x,dist+1))
        out_rdd = out_rdd.union(n_RDD)
        out_rdd = out_rdd.reduceByKey(lambda x,y:min(x,y))
        it_count += 1
    return out_rdd

