import pyspark
from P5_bfs import BFS_Graph

sc = pyspark.SparkContext('local','BFS_Graph')

links=sc.textFile('links-simple-sorted.txt')
page_names=sc.textFile('titles-sorted.txt')

# function for creating the graph
def mapToKV(link):
    src, dests = link.split(': ')
    dests=[int(x) for x in dests.split(' ')]
    return (int(src), dests)

# get graph for links, where the value is a list of children
graph=links.map(mapToKV)

# for finding the corresponding link for a page
IndPage=page_names.zipWithIndex()
IndPage=IndPage.map(lambda (page,id): (page,id+1))

#for finding the corresponding page for a link
PageInd=IndPage.map(lambda (page,id): (id, page))

#startPt=IndPage.lookup('Harvard')[0]
#endPt=IndPage.lookup('Kevin_Bacon')[0]
startPt=IndPage.lookup('Kevin_Bacon')[0]
endPt=IndPage.lookup('Harvard')[0]

SG=BFS_Graph(graph,startPt,endPt)

parents=[PageInd.lookup(endPt)[0]]
(p,dests,d)=SG.lookup(endPt)[0]
i=0
while i<d-1:
    str=PageInd.lookup(p)[0]
    parents.append(str)
    (p,dests,d)=SG.lookup(p)[0]
    i=i+1
(p,dests,d)=SG.lookup(p)[0]
str=PageInd.lookup(p)[0]
parents.append(str)

print parents
