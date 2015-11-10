import pyspark
from pyspark import SparkContext, SparkConf
sc = SparkContext()

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
# links  = sc.textFile('links-simple-sorted.txt')
# page_names = sc.textFile('titles-sorted.txt')



#pair names with their indexes. 
#notice that indexes are shifted to right by 1
#also swap the position of name and index
index_names = page_names.zipWithIndex().map(lambda tpl: (tpl[1]+1, tpl[0]))

names_index = index_names.map(lambda tpl: (tpl[1],tpl[0]))

def link_string_to_KV(s):
    name, indexes = s.split(': ')
    indexes = [int(index) for index in indexes.split(' ')]
    return (int(name), indexes)
links_final = links.map(link_string_to_KV)

def KV_further(tpl):
    return (tpl[0], (10**6, tpl[0], tpl[1]))
links_further = links_final.map(KV_further)





def combine_dist_parent_with_children(a,b):
    '''we have nodes like (B, (1,A,[]) ). Also we have (B, (10**6,B,[neighbors of B]) ).
       this function combines these information together.
       Here a is with form (1,A,[]) and b is with form (10**6,B,[neighbors of B])'''
    a_dist = a[0]
    b_dist = b[0]

    if a_dist < b_dist:
        return (a_dist, a[1], [set(a[2]+b[2])]) 
    else:
        return (b_dist, b[1], [set(a[2]+b[2])]) 



def set_starting_char_dist(node, start):
    '''used at the beginning of bfs.
       it finds the node that we start bfs with and set its distance to 0.'''
    if node[0] == start:
        #set dist=0 if node is the starting_char node
        return (node[0],(0, node[0], node[1][2]))
        
    else:
        return node



def set_parent_and_dist(node, n):
    '''getting in a node with form (char, (dist, parent, [children]))
       return a list of (child, (dist+1, char, [])) + node itself'''
    children = node[1][2]
    #parent = node[1][1] #just for information, did not use
    dist = node[1][0]
    self = node[0]
    to_return = [node]
    
    if dist == n:
        for child in children:
            new = (child, (dist+1, self, []))
            to_return.append(new)
        #don't forget to append the node that we are working with

    return to_return



def bfs_new(starting_char, end_char, graph):
    dist_traversed = 0
    #set the starting_char node dist = 0
    graph = graph.map(lambda x: (set_starting_char_dist(x, starting_char)) )

    while True:
        graph = graph.flatMap(lambda y: (set_parent_and_dist(y, dist_traversed)) )
        #after we update the distances of children, we "step" further
        dist_traversed += 1
        graph = graph.reduceByKey(lambda a,b: (combine_dist_parent_with_children(a,b)))
        
        #check if end_char has been touched
        #if so, bfs has been done
        check_dist = graph.lookup(end_char)[0][0]
        if check_dist < 10**6:
            return (graph, dist_traversed)





def trace_path(start_name, target_name, graph):
    start_num = names_index.lookup(start_name)[0]
    target_num = names_index.lookup(target_name)[0]
    g_rdd = bfs_new(start_num, target_num, graph)[0]
    
    length = g_rdd.lookup(target_num)[0][0]
    path_list = [target_num]
    for i in length:
        parent = g_rdd.lookup(target_num)[0][1]
        path_list = [parent] + path_list
        target_num = parent
    
    path_list_name = []
    for j in path_list:
        name = index_names.lookup(j)[0]
        path_list_name.append(name)
    return path_list_name

 



H_to_K = trace_path('Harvard University','Kevin Bacon',links_further)
K_to_H = trace_path('Kevin Bacon','Harvard University',links_further)
myfile = open('P5.txt', 'w')
myfile.write(H_to_K)
myfile.write(K_to_H)
myfile.close()       