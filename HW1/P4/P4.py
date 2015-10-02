import findspark
findspark.init()
import pyspark
import itertools


if __name__ == "__main__":  
    sc = pyspark.SparkContext()

    # Load the lines of the text file and switch the order of the hero, issue so that I can use
    #    the issue as the key
    source = sc.textFile("source.csv").map(lambda x: x.split(',')).map(lambda x: (x[1], x[0]))

    # Group by issue, and create tuples of (hero, all other heros in that issue) for that issue.
    #     FlatMap to take all the tuples out of being in lists by issue.  
    source1 = source.groupByKey().mapValues(lambda heros: [(hero1, [hero2 for hero2 in heros if hero2 != hero1]) for hero1 in heros]).values().flatMap(lambda x: x)

    # Define a function to take all the lists that are returned by reduceByKey and flatten them,
    #    then use list(set()) to take only unique values
    def func(*args): return list(set(itertools.chain(*args)))

    # Group by each superhero as a key, and take only unique edges
    source2 = source1.reduceByKey(func)

    ###def source(): return source2
