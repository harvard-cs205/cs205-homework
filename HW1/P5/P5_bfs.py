
import pyspark
from pyspark import SparkContext
sc = SparkContext()


# read in data
#links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
#page = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

links = sc.textFile('links-simple-sorted.txt')
page = sc.textFile('titles-sorted.txt')

# construct rdd of all neighbors
rdd1 = links.map(lambda x: x.split(': '))
def get_dest(s):
    return [int(x) for x in s.split(' ')]
rdd2 = rdd1.map(lambda x: (int(x[0]), get_dest(x[1])))

# partition the graph rdd
rdd_link = rdd2.partitionBy(256).cache()

# set index to page names and get Kevin Bacon, Harvard index
rdd3 = page.zipWithIndex()
rdd_page = rdd3.map(lambda x: (x[1]+1, x[0])).cache()

# get index of start and end nodes
start = rdd_page.filter(lambda x: x[1] == 'Kevin_Bacon').take(1)[0][0]
end = rdd_page.filter(lambda x: x[1] == 'Harvard_University').take(1)[0][0]


def bfs(start, end, rdd_link, rdd_page):
    find_path = sc.accumulator(0)
    find_path.add(1)
    # keep track of path
    path = []
    # keep track of current layer of nodes
    current = [start]
    l = [start]
    index = 1


    while find_path.value != 0:
        find_path.value = 0
        this_layer = rdd_link.filter(lambda x: x[0] in current).flatMap(lambda x: x[1]).distinct().map(lambda x: (index,x))
        path = path + this_layer.collect()
        current = list(set(this_layer.values().collect()) - set(l))

        # update nodes visited: l                                                                                                 
        l = l + current
        index = index + 1

        # check if end node is found                                                                                       
        if (end in current == False):
            find_path.add(1)


    final_path = []
    final_path.append(rdd_page.lookup(end)[0])
    while index > 1:
            index = index - 1
            this_path = rdd_link.filter(lambda x: end in x[1]).map(lambda x: (index,x[0])).collect()
            right_path = list(set(this_path).intersection(set(path)))
            end = right_path[0][1]
            final_path.append(rdd_page.lookup(end)[0])
    final_path.append(rdd_page.lookup(start)[0])
    final_path.reverse()

    print final_path


bfs(start, end, rdd_link, rdd_page)






 


