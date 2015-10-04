from pyspark import SparkContext
sc = SparkContext()

sc.setLogLevel('WARN')

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

neighbor_graph = links.map(link_string_to_KV)
page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()
neighbor_graph = neighbor_graph.partitionBy(256).cache()

def rank(x):
    a = page_names.filter(lambda (K, V): V == x).collect()
    return a[0][0]

def name(x):
    a = page_names.filter(lambda (K, V): K == x).collect()
    return a[0][1]

def bfs(x,y,rdd):    
    keys=[rank(x)]
    keys_backup=[rank(x)]
    back=rank(y)
    go_path=[]
    i=0
    while (back in keys) == False:
        i+=1
        go_rdd=rdd.filter(lambda x: x[0] in keys).flatMap(lambda x: x[1]).distinct().map(lambda x: (i,x))
        go_path+=go_rdd.collect()
        keys=set(go_rdd.values().collect()).difference(set(keys_backup))
        keys_backup+=set(keys)
        if len(keys)==0:
            i=-1
            break        
    if i==-1:
        print "Cannot Reach"
    else:
        node={}
        node[0]=x
        node[i]=y
        while i>1:
            i-=1
            back_path=rdd.filter(lambda x: back in x[1]).map(lambda x: (i,x[0])).collect()
            go_back=list(set(back_path).intersection(set(go_path)))
            back=go_back[0][1]   
            node[i]=name(back)
        print node

bfs("Kevin_Bacon","Harvard_University",neighbor_graph)
bfs("Harvard_University","Kevin_Bacon",neighbor_graph)
