from pyspark import SparkContext
from P5_bfs import BFS

def get_all_link_pairs(link_str):
    """ Takes a string formatted as provided by Wikipedia, i.e.:
    "[page_number]: first_link second_link ... last_link"
    and returns a tuple of ((page_number, first_link), ... (page_number, last_link))
    To be used with flatMap to get all edges in the graph.
    """

    # The page number is followed by a colon, so splitting by a colon
    # Gives us easy access to it
    page_number = link_str.split(':')[0]

    # Everything else is separated by spaces, so we can take all of that
    list_of_links = link_str.split()[1:]

    # Store our results
    results = []
    for other_page in list_of_links:
        # Note that this preserves directionality of the graph
        results.append((page_number, other_page))

    return tuple(results)                                 # flatMap expects a tuple

if __name__ == "__main__":

    # First get a SparkContext object
    sc = SparkContext(appName='shortest_paths')

    # Get the links for the wikipedia pages
    links = sc.textFile('/home/nick//Documents/links-simple-sorted.txt')

    # Now load in the pages 
    page_names = sc.textFile('/home/nick/Documents/titles-sorted.txt')

    # Get the pages with their indices and correct for 1-indexing
    # Our graph will be represented as a list of (page_num, [linked_page_num_1, ...])
    # So we will need to look up the pages by name in this RDD to find out their page number
    # for use in the actual graph
    pages_with_indices = page_names.zipWithIndex().map(lambda (x, y): (x, y+1))

    # Get all pairs of links
    all_link_pairs = links.flatMap(get_all_link_pairs)

    # Now we have the pages as keys - with the individual connections as values 
    # Thus reduceByKey gets us to an adjacency list representation
    # First wrap everything up in a list so we can just add
    page_graph = all_link_pairs.map(lambda (x, y): (x, [y]))
    page_graph = page_graph.reduceByKey(lambda x, y: x + y)

    # Find the distance from Kevin_Bacon to Harvard_University and the reverse
    harvard_id = str(pages_with_indices.lookup("Harvard_University"))
    bacon_id = str(pages_with_indices.lookup("Kevin_Bacon"))
    
    # Do the searches
    harvard_path_graph = BFS(page_graph, harvard_id, sc)
    bacon_path_graph = BFS(page_graph, bacon_id, sc)


    print 'Harvard id and Bacon id:', harvard_id, bacon_id

    # Actually look them up
    harv_bacon_dist = harvard_path_graph.lookup(bacon_id)
    bacon_harv_dist = bacon_path_graph.lookup(harvard_id)

    with open('output.txt','w') as out_file:
        print >> out_file, "Distance from Harvard to Bacon:", harv_bacon_dist
        print >> out_file, "Distance from Bacon to Harvard:", bacon_harv_dist
