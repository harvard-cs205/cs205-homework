# Script for finding shortest paths in the wiki data (suitable for AWS)

# Compute shortest paths from "Kevin_Bacon" to "Harvard_University" and the reverse.

def linkStringToKV(s):
	src, dests = s.split(': ')
    dests = [int(dest) for dest in dests.split(' ')]
    return (int(src), dests)

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)
neighbor_graph = links.map(linkStringToKV)

# Eventually run
# spark-submit --master yarn-client --num-executors 2 --executor-cores 4 --executor-memory 5g P5_bfs.py