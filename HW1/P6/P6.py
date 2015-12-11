import numpy as np
from functools import partial
import findspark
findspark.init()
import pyspark


def pre_process_data( filename, sc ):
    wlist = sc.textFile( filename )
    wlist_words = wlist.flatMap( lambda x: x.split() )
    wlist_words_filter = wlist_words.filter( lambda x: not( x.isdigit() or x.isupper() or ( x[:-1].isupper() and x[-1:] == "." ) ) )
    wlist_words_index = wlist_words_filter.zipWithIndex().map( lambda (x,y): (y,x) )
    wlist_words_index_2 = wlist_words_index
    size = wlist_words_index_2.count()
    wlist_words_index_2 = wlist_words_index_2.map( lambda ( x, y ): ( x-1, y ) if x != 0 else ( size, y ) )
    wlist_words_2 = wlist_words_index.join( wlist_words_index_2 )
    wlist_words_index_2 = wlist_words_index_2.map( lambda ( x, y ): ( x-1, y ) if x != 0 else ( size, y ) )
    wlist_words_final = wlist_words_2.join( wlist_words_index_2 ).map( lambda ( x, y ): ( ( y[0][0], y[0][1], y[1] ) , 1 ) ).reduceByKey( lambda a,b: a + b ).map( lambda ( x, y ): ( ( x[0], x[1] ), [ ( x[2], y ) ] ) ).reduceByKey( lambda a,b: a + b)
    wlist_words_final.count()
    wlist_words_final.cache()
    return wlist_words_final

    
def markov_prediction_third_word( ( word1, word2 ), wlist_words_final_dict ): 
    transition_prob = wlist_words_final_dict[ ( word1, word2 ) ]
    normalizing_constant = np.array( [ i[1] for i in transition_prob ] ).sum()
    transition_prob_final = { key: ( float( value )/normalizing_constant ) for (key, value) in transition_prob }
    unif = np.random.uniform(0,1)
    transition_prob_final_sorted_keys = sorted( transition_prob_final )
    cdf = 0
    for key in transition_prob_final_sorted_keys:
        cdf = transition_prob_final[ key ] + cdf
        if unif < cdf:
            return key


def main():
    sentence_length = 20
    sentence_count = 10
    sc = pyspark.SparkContext()
    
    wlist_words_final = pre_process_data( "Shakespeare.txt", sc )
    wlist_words_final_dict = { key: value for (key, value) in wlist_words_final.collect() }

    #broadcast_wlist_words_final = sc.broadcast( wlist_words_final_dict )
    
    sentences = wlist_words_final.takeSample( True, sentence_count )
    sentences = sc.parallelize( sentences )
    count = 0
    while( count < sentence_length - 2 ):
        sentences = sentences.map( lambda (x,y): ( list(x), markov_prediction_third_word( ( x[len(x)-2], x[len(x)-1] ), wlist_words_final_dict ) ) ).map( lambda ( x, y ): ( x + [y], [] ) )
        sentences.cache()
        count = count + 1

    print sentences.count()
    print sentences.take(10)

if __name__ == "__main__": main()

