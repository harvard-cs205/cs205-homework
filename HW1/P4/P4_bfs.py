import numpy as np
from itertools import chain

def P4_bfs(grph,source):
	
	nodesEdges = [source]
	searchedDict = {}
	[searchedDict.setdefault(hero, []).append(0) for hero in nodesEdges]
	for i_dist in xrange(1,10):
		
		if len(nodesEdges) == 0:
			break
		
		exaNodeSet = grph.filter(lambda KV: KV[0] in nodesEdges).flatMap(lambda x: x[1]).collect()
		nodesEdges = list(set(exaNodeSet).difference(set([key for key in searchedDict.keys()])))
		[searchedDict.setdefault(hero, []).append(i_dist) for hero in nodesEdges]    	
		

		
	
	print "Exhausted graph at a distance of:", i_dist
	print "With "+source+" as your source, you touched "+ str(len(searchedDict)) + " nodes."
	
	return len(searchedDict), searchedDict
		

