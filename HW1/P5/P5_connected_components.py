# This code is correct, returning an rdd of distinct labels for every
# connected component in the graph (try it with small examples!)
# Alas, it takes a very long time on AWS...
def connected_components(adj_list_rdd):
  num_parts = 64
  labels_rdd = adj_list_rdd.map(lambda (k,v): (k,k)).partitionBy(num_parts).cache()
  last = labels_rdd.values().countByValue()
  adj_list_rdd.cache()
  iteri = 0
  while 1:
    print iteri
    iteri += 1
    combined = adj_list_rdd.join(labels_rdd)
    combined = combined.flatMap(lambda (k, (neighbors, label)): ([(n, label) for n in neighbors]))
    tbr = labels_rdd.cogroup(combined)
    labels_rdd = tbr.map(lambda (k, (v1, v2)): (k, min(list(v1)[0], list(v2)[0])))
    if labels_rdd.values().countByValue() == last:
      break
    last = labels_rdd.countByValue()
  return labels_rdd