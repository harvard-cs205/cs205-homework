from P5_bfs import *

neighbor_pair=neighbor_graph.flatMap(lambda x: [(x[0],x[1][j]) for j in range(len(x[1]))]).cache()

import copy
keys_backup=[]
remain_keys=set(neighbor_graph.keys().collect())
last_keys=[]
i=0
while remain_keys!=set([]):
    i+=1
    keys=neighbor_graph.filter(lambda x: (x[0] in remain_keys)==1).keys().take(1)
    keys_backup+=keys
    while keys!=set([]):
        right=neighbor_pair.filter(lambda x: x[0] in keys).map(lambda x: x[1]).distinct().collect()
        left=neighbor_pair.filter(lambda x: x[1] in keys).map(lambda x: x[0]).distinct().collect()
        union=set(right).union(set(left))
        keys=union.difference(set(keys_backup))
        keys_backup+=set(keys)
    remain_keys=set(remain_keys)-set(keys_backup)
    last_keys=set(keys_backup)-set(last_keys)
    print last_keys
    last_keys=copy.copy(keys_backup)

print i
