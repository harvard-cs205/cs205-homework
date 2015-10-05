import pyspark

sc = pyspark.SparkContext()
sc.setLogLevel('WARN')



def extractNodes(x):
    global globallevel
    level = globallevel
    if x[1][1] == level:# only look at current level nodes
        flattened = [x]
        for nei in x[1][0]:
            flattened += [(nei, ([], level+1, x[0]))]# create a new line of neighbor with correct level and origin
        return flattened
    else:
        return [x]

def refreshNodeComb(list_attr):
    global globallevel
    level = globallevel
    for i in list_attr:
        if i[1] <level+1:
            return i
    flag = [False, False]
    for i in list_attr:
        if i[2]==-1:
            neighbor = i[0]
            flag[0] = True
        if len(i[0]) == 0 and i[2]!=-1:
            origin = i[2]
            level = i[1]
            flag[1] = True
        if flag[0] and flag[1]:
            return (neighbor, level, origin)
    return list_attr[0]


def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)



##################################################
# Reading files
##################################################

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
#links = sc.textFile('/Users/haosutang/links-simple-sorted.txt')
#page_names = sc.textFile('/Users/haosutang/titles-sorted.txt')
neighbor_graph = links.map(link_string_to_KV)
page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()

# find Kevin Bacon
Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
# This should be [(node_id, 'Kevin_Bacon')]
assert len(Kevin_Bacon) == 1
Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id

# find Harvard University
Harvard_University = page_names.filter(lambda (K, V):
                                       V == 'Harvard_University').collect()
# This should be [(node_id, 'Harvard_University')]
assert len(Harvard_University) == 1
Harvard_University = Harvard_University[0][0]  # extract node id

neighbor = neighbor_graph.cache()





start = Kevin_Bacon
end = Harvard_University
max_iteration=10
accum = sc.accumulator(-1)
globallevel = accum.value
#add a level flag to the end, never visited is max_iteration, current
#also add a origin mark, start node is 0, never visited is -1


##################################################
# Find the tree containing the path
##################################################
mapped = neighbor.map(lambda l: (l[0], (l[1], max_iteration, -1)) if l[0]!=start else (l[0], (l[1], globallevel, 0)))

tree = None

while globallevel<max_iteration:
    accum.add(1)
    globallevel = accum.value
    print 'global level', globallevel

    transition = mapped.flatMap(extractNodes)

    mapped = transition.groupByKey().map(lambda l: (l[0], refreshNodeComb(list(l[1]))))

    if not mapped.filter(lambda(k,v): k==end and v[1]!=max_iteration).isEmpty():# if visited
        print '**********************************************'
        print '******************Found!**********************'
        print '**********************************************'
        tree = mapped.filter(lambda (k,v): v[1]!=max_iteration and (k==end or v[1]<=globallevel))
        break
    mapped.persist()

##################################################
# Trace back the tree to find a path
##################################################
assert tree!=None
path = [end]
key=end
while key!=start:
    origin = tree.filter(lambda (k,v): k ==key).collect()[0][1][2]
    path += [origin]
    tree = tree.filter(lambda (k,v): k==origin or v[1]<=globallevel)
    tree.persist()
    key = origin
    globallevel -= 1
    

shortest_path_id = []

for i in path:
	shortest_path_id += [page_names.filter(lambda (K, V): K==i).collect()]

print '********************PATH**********************'
print list(reversed(shortest_path_id))
print '**********************************************'