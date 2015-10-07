import findspark
findspark.init()
import pyspark

def main():
    sc = pyspark.SparkContext()
    wlist = sc.wholeTextFiles("./CSV Format")
    combined_wlist = wlist.flatMap( lambda (a,b): b.split() )
    combined_wlist_sort = combined_wlist.map( lambda x: ( "".join( sorted( x ) ), [x] ) )
    combined_wlist_sort_reduce = combined_wlist_sort.reduceByKey( lambda x,y: x + y )
    combined_wlist_sort_reduce_count = combined_wlist_sort_reduce.map( lambda (x,y): (x, [y]) if ( type(y) is not list ) else (x, 
    y) ).map( lambda (x,y): ( x, len(y), y) )

    wlist_check = combined_wlist_sort_reduce_count.takeOrdered( 1, lambda (x,y,z): -y )
    print wlist_check
    
    
if __name__ == "__main__": main()
    
