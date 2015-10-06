def main():
	links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
	page_names = sc.textFile('s3 :// Harvard-CS205/wikipedia/titles-sorted.txt')
	indexed_page_names=page_names.zipWithIndex()
	# Put into (num,(num,num,...)) format
	links=links.map(lambda x: x.split(':')).mapValues(lambda x:x.strip().split(' '))
	accum = sc.accumulator(0)
	accum.value=0
	links=BFS(links,indexed_page_names,'Harvard_University', '', accum)

def BFS(links, page_names,source_node, dest_node,accum): 
    # get indicies for source and destination node
    source_node=page_names.lookup(source_node)[0]
    dest_node=page_names.lookup(dest_node)[0]
    num_to_names=page_names.map(lambda x: (x[1],x[0]))
    # chars list will include the node, distance to that node, and the parent node
    chars_list = sc.parallelize([(source_node,(0,source_node))])
    new_chars = sc.parallelize([source_node])
    print chars_list.collect()
    path= [dest_node]
    # same logic as used for BFS in problem 4
    while not new_chars.isEmpty():
        level=accum.value
        new_chars=chars_list.filter(lambda x: x[1][0]==level).join(links).map(lambda x:[(a,(x[1][0][0]+1,x[0])) for a in x[1][1]]).flatMap(lambda x:x).distinct()
        accum.add(1)
        chars_list=chars_list.union(new_chars).groupByKey().mapValues(lambda x: list(x)[0])
        # if you find a match
        if chars_list.lookup(dest_node) != []:
            parent=chars_list.lookup(dest_node)[0][1]
            # append the parent node
            path.append(parent)
            # back track until you get to the source node
            while parent != source_node:
                parent=chars_list.lookup(parent)[0][1]
                path.append(parent)
                #reverse the list, and then look up the names of the nodes 
            return [num_to_names.lookup(a) for a in list(reversed(path))]
    return -1