import pyspark
from pyspark import SparkContext
sc = SparkContext()


def reduction((p1,x1,d1),(p2,x2,d2)):
    if d1<d2: 
    	return (p1,list(set(x1+x2)),d1)
	else: 
		return (p2,list(set(x1+x2)),d2)
   
def find_comp(page2link,bfs_graph,id):
	parents=[page2link.lookup(id)[0]]
	i=0
	while True:
		(p,x,d)=bfs_graph.lookup(id)[0]
	    str=page2link.lookup(p)[0]
	    parents.append(str)
	    i +=1
	    if i<d-1: break
	parents.append(page2link.lookup(bfs_graph.lookup(p)[0][0])[0])
	return parents


def bfs(graph,HUID,BID):
	def compute_dist((src,(p,x,d))):
    res=[(src,(p,x,d))]
    if d==flag:
        for xx in x:
            res.append((xx,(src,[],d+1)))
    return res

    bfs_graph=graph.map(lambda yy: (yy[0],(yy[0],yy[1],0)) if yy[0]==HUID else (yy[0],(yy[0],yy[1],1000)))
    flag=0
    while True:
        bfs_graph=bfs_graph.flatMap(compute_dist)
        bfs_graph=bfs_graph.reduceByKey(reduction)
        
        if len(bfs_graph.filter(lambda bg: bg[0]==BID and bg[1][2]<1000).collect())==1: 
        	break
        else: flag+=1

    return bfs_graph


#links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
#page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
links = sc.textFile('links-simple-sorted.txt')
page_names = sc.textFile('titles-sorted.txt')

link1 = links.map(lambda x: x.split(': '))
graph =link1.map(lambda x: (x[0],x[1])).mapValues(lambda x: x.split(" ")).flatMapValues(lambda x: x).map(lambda x: (int(x[0]),int(x[1]))).partitionBy(128).cache()

# link to page and page to link
link2page=page_names.zipWithIndex().map(lambda (x, y): (x,y+1))
page2link=lpage_names.zipWithIndex().map(lambda (x, y): (y+1, x))

HUID=link2page.filter(lambda (x,y): x == 'Harvard_University').collect()[0][0]
BID=link2page.filter(lambda (x,y): x == 'Kevin_Bacon').collect()[0][0]

bfs_graph=bfs(graph,HUID,BID)
comp = find_comp(page2link,bfs_graph,BID)

print comp

