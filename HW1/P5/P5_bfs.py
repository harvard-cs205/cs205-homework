import pyspark
from pyspark import SparkContext, SparkConf
sc = SparkContext()

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
# links  = sc.textFile('links-simple-sorted.txt')
# page_names = sc.textFile('titles-sorted.txt')

def link_string_to_KV(s):
    name, indexes = s.split(': ')
    indexes = [int(index) for index in indexes.split(' ')]
    return (int(name), indexes)
links_final = links.map(link_string_to_KV)



##########functions for treating links##########
def bfs(graph, starting_char):
    char_list = [starting_char]
    BFS_list  = []
    
    while len(char_list)!=0:
        neighbors = graph.filter(lambda tpl: tpl[0] in char_list).flatMap(lambda tpl: tpl[1]).collect()
        BFS_list  = BFS_list + char_list
        char_list = []
        char_list = char_list + list(set(neighbors)-set(BFS_list))                                     
        print len(list(set(neighbors)-set(BFS_list)))
    #print len(BFS_list)
    


def bfs_new(graph, starting_char, end_char):
    char_list = [(starting_char,starting_char)]
    BFS_list  = []
    
    while len(char_list)!=0:
        neighbors = graph.filter(lambda tpl: tpl[0] in [c for (p,c) in char_list]).flatMap(lambda tpl: [(key,value) for key, value in zip([tpl[0]]*len(tpl[1]), tpl[1])] ).collect()
        BFS_list  = BFS_list + char_list   #all visited nodes
        char_list = []
        children = [c for (p,c) in BFS_list]
        for tpl in neighbors:
        	if tpl[1] == end_char:
        		char_list.append(tpl)
        		BFS_list += char_list
        		return #trace_path(BFS_list, end_char, starting_char)
        	if tpl[1] not in children:
        		char_list.append(tpl)
    return []


        
def trace_path(BFS_list, target, start):
	result = [target]
	while (target != start):
		for tpl in BFS_list:
			if tpl[1] == target:
				target = tpl[0]
				result = [target] + result
				break
	return result    
##########end of functions for treating links##########



#pair names with their indexes. 
#notice that indexes are shifted to right by 1
#also swap the position of name and index
names_index = page_names.zipWithIndex().map(lambda tpl: (tpl[1]+1, tpl[0]))



##########functions for treating page_names##########
def name_to_index(name_string, rdd):
    return rdd.filter(lambda x: x[1]==name_string).collect()[0][0]



def index_to_name(index, rdd):
    return rdd.filter(lambda x: x[0]==index).collect()[0][1]



def print_sequential_path(path_list, rdd):
    result = []
    for i in path_list:
        result.append(index_to_name(i, rdd))
    return result
##########end of functions for treating page_names##########



Harvard_University_index = name_to_index('Harvard_University', names_index)
Kevin_Bacon_index = name_to_index('Kevin_Bacon', names_index)

path_list_H_to_K = bfs_new(links_final, Harvard_University_index, Kevin_Bacon_index)
path_list_K_to_H = bfs_new(links_final, Kevin_Bacon_index, Harvard_University_index)

H_to_K = print_sequential_path(path_list_H_to_K, names_index)
K_to_H = print_sequential_path(path_list_K_to_H, names_index)



print 'path from Harvard_University to Kevin_Bacon:', H_to_K
print 'path from Kevin_Bacon to Harvard_University:', K_to_H
