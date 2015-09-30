import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()

def find_anagrams(data,master):
    '''Gets all permutations of word, finds valid cases by comparing against master word list'''
    from itertools import permutations
    perms = set([''.join(p) for p in permutations(data[0])])    # compute permutations
    valid_perms = list(perms.intersection(master))              # select matches against master word list
    return data + (valid_perms,)                                # return (<string>,<ct>,[<match1>,<match2>,...])

rdd = sc.textFile('EOWL_words.txt')

find_anagrams(
    (rdd.map(sorted)                            # sort characters in each word                           
    .map(lambda x: (''.join(x),1))              # join sorted characters, return (<sorted>,ct=1) 
    .reduceByKey(lambda a,b: a+b)               # sum key counts (ie. find number of anagrams) 
    .takeOrdered(1, key=lambda x: -x[1])[0]),    # return word with most anagrams (ordered by ct)
    master=rdd.collect()
)