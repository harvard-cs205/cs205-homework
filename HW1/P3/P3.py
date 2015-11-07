import findspark
findspark.init()
import pyspark


if __name__ == "__main__":  
    sc = pyspark.SparkContext()
    wlist = sc.textFile("EOWL_words.txt")

    # Define a function to return tuples of the sorted words and their original spellings
    def func(a):
        return (''.join(sorted(a)), a)

    # Map the above function to the word list
    wlist_pairs = wlist.map(func)

    # Group by the sorted name, and put the words with the same sorted name (the anagrams) into a list
    grouped_pairs_lists = wlist_pairs.groupByKey().mapValues(list)
    
    
    # Take the pairs of sorted names and anagrams, and return the sorted name, the number of anagrams, and the 
    #     list of anagrams
    grouped_pairs = grouped_pairs_lists.map(lambda (x,y): (x, len(y), y))

    # Sort the grouped pairs by the number of anagrams, and print the first 
    print grouped_pairs.sortBy(lambda x: x[1], ascending=False).first()
