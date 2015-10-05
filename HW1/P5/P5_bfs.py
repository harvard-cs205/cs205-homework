import pyspark

def BFS_Graph(graph,startPt,endPt):
    currLevel=0
    
    def DistrFunc((src,(p,dests,d))):
        listChildren=dests
        result=[(src,(p,dests,d))]
        if d==currLevel:
            for child in listChildren:
                result.append((child,(src,[],d+1)))
        return result

    def reduceFunc((p1,dests1,d1),(p2,dests2,d2)):
        dist1=d1
        children1=dests1
        dist2=d2
        children2=dests2
        children=[]
        if dist1<dist2:
            d=dist1
            p=p1
        else:
            d=dist2
            p=p2
        children=list(set(children1+children2))
        return (p,children,d)

    # map each node in graph rdd to tuple
    SG=graph.map(lambda node: (node[0],(node[0],node[1],0)) if node[0]==startPt else (node[0],(node[0],node[1],1000)))
    while True:
        SG=SG.flatMap(DistrFunc)
        SG=SG.reduceByKey(reduceFunc)
        tmp=SG.filter(lambda sg: sg[0]==endPt and sg[1][2]<1000).collect()
        if len(tmp)==1:
            break
        else:
            currLevel=currLevel+1
            print currLevel

    return SG
