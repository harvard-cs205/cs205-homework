import pyspark

if __name__ == '__main__':
  sc = pyspark.SparkContext(appName='YK-P3')
  wlist = sc.textFile('words.txt')

  sortedWords = wlist.map(lambda x: (''.join(sorted(x)), set([x])))
  mergedWords = sortedWords.reduceByKey(lambda s1, s2: s1 | s2)
  finalRdd = mergedWords.map(lambda x: (x[0], len(x[1]),list(x[1])))
  finalRdd.cache()

  maxlen = 0
  anagram = None
  for item in finalRdd.collect():
    _, length, anag = item
    if length > maxlen:
      maxlen = length
      anagram = anag

  with open('P3.txt', 'w') as fh:
    fh.write(str(maxlen) + '\n')
    fh.write(str(anagram))
  

