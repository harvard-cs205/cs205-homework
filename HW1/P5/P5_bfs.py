# Caution: if src is not connected to dest, this runs forever
def find_all_shortest_paths(src, dest, graph, sc):
  accum = sc.accumulator(0)
  num_parts = 64
  graph = graph.partitionBy(num_parts).cache()
  distances = sc.parallelize([(src, [src])]).partitionBy(num_parts)
  while 1:
    accum.value = 0
    print "new iteration"
    assert(distances.partitioner == graph.partitioner)
    distances = distances.join(graph).map(lambda (k, v): (v[1], v[0] + [v[1]])).partitionBy(num_parts)
    test = distances.filter(lambda (k, v): k == dest)
    test.foreach(lambda x: accum.add(1))
    if accum.value > 0:
      break
  return test.map(lambda (k,v): v)