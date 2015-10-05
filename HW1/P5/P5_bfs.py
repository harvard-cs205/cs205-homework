import findspark
findspark.init('/Users/Grace/spark-1.5.0-bin-hadoop2.6/')
import pyspark

def combine_nodes((distance1, parent1, children1), (distance2, parent2, children2)):
    if distance1 < distance2 :
        return (distance1, parent1, list(set(children1+children2)))
    else :
        return (distance2, parent2, list(set(children1+children2)))

def update_node(x, level): #x is (node, (distance, parent, children))
    (node, (distance, parent, children)) = x
    return_list = [x]

    if distance == level:
        for child in children:
            return_list.append((child, (level+1, node, [])))

    return return_list


def finding_directed_paths(graph, start_node, end_node):
    INFINITY = 10**8
    level = 0
    dist_graph = graph.map(lambda (node, children): (node, (0, node, children)) if (node == start_node) else (node, (INFINITY, node, children)))

    while True :
        end_node_distance = dist_graph.lookup(end_node)[0][0]
        if end_node_distance < INFINITY : #(end_node, (INFINITY, parent, [neighbors]))
            break
        else :
            dist_graph = dist_graph.flatMap(lambda x: update_node(x, level))
            dist_graph = dist_graph.reduceByKey(combine_nodes)
            level+=1

    return dist_graph

def path_in_index(graph, start_index, end_index): #node
    path = [end_index]
    while True:
        if start_index in path:
            break
        else:
            parent = graph.lookup(path[-1])[0][1]
            path.append(parent)
    path.reverse()
    return path


def print_path(name_index, path_in_index):
    path = []
    for index in path_in_index:
        entry = name_index.filter(lambda (x, y):y==index)
        name = entry.collect()[0][0]
        path.append(name)
    return path




sc = pyspark.SparkContext()
links = sc.textFile('links-simple-sorted.txt')
page_names = sc.textFile('titles-sorted.txt')

name_index = page_names.zipWithIndex().map(lambda (page_name, index): (page_name, unicode(index+1))) #(harvard, index)
name_index.cache()
#print name_index.take(3)

index_links = links.map(lambda line : tuple(line.split(':')))
list_links = index_links.map(lambda (index, unicode_links) : (index, unicode_links.split())) #(index, [link1, link2, ...])
#print list_links.take(2)

harvard_index = name_index.lookup('Harvard_University')[0]
#print harvard_index
kevin_index = name_index.lookup('Kevin_Bacon')[0]
#print kevin_index



################################
# Harvard -> Kevin
###############################
answer_graph = finding_directed_paths(list_links, harvard_index, kevin_index)
#answer_graph.take(1)
answer_graph.cache()
#print answer_graph.take(1)

#print_path(name_index, [u'2152782', u'4530382', u'44868', u'2729536'])
#print path_in_index(answer_graph, harvard_index, kevin_index)
print print_path(name_index, path_in_index(answer_graph, harvard_index, kevin_index))



################################
# Kevin -> Harvard
###############################
kevin_graph = finding_directed_paths(list_links, kevin_index, harvard_index)
kevin_graph.cache()
#print kevin_graph.take(1)

print print_path(name_index, path_in_index(kevin_graph, kevin_index, harvard_index))
#path_in_index(kevin_graph, kevin_index, harvard_index)


################################
# Interesting Pair : Korea -> Harvard
###############################
korea_index = name_index.lookup('Korea')[0]

korea_graph = finding_directed_paths(list_links, korea_index, harvard_index)
korea_graph.cache()
#print korea_graph.take(1)

#path_in_index(korea_graph, korea_index, harvard_index)
#print_path(name_index, [u'2784007', u'3829551', u'2152782'])
print print_path(name_index, path_in_index(korea_graph, korea_index, harvard_index))