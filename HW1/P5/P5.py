from P5_bfs import *
sc.setLogLevel('ERROR')

# gets info
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

# clean up to work
links = links.map(lambda x: x.split(": "))
links = links.map(lambda (page, links) : (page, links.split(" ")))
page_names = page_names.zipWithIndex().mapValues(lambda x : x+1)

# save harvard and bacon
harvard = page_names.lookup("Harvard University")[0]
bacon = page_names.lookup("Kevin_Bacon")[0]


bfs(harvard, bacon, links, sc)
bfs(bacon, harvard, links, sc)

