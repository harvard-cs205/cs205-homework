SG=hero_graph.map(lambda node: (node[0], node[1], -1))
# there's only one element in the target_node
startingNode=SG.filter(lambda node: node[0]==v_str).collect()
#print Current_Nodes
Diameter=10
iter=0
current_neighbor=startingNode[0][1]
while iter<Diameter:
    tmplist=[]
    SG=SG.map(lambda node: (node[0],node[1],1) if node[0] in current_neighbor and node[2]!=-1 else node)
    tmp=SG.filter(lambda node: node[0] in current_neighbor and node[2]==-1).collect()
    numNeighbor=len(tmp)
    print numNeighbor
    i=0
    while i<numNeighbor:
        tmplist.extend(tmp[i][1])
        i=i+1
    current_neighbor=tmplist
    iter=iter+1

r=SG.filter(lambda node: node[2]==1)