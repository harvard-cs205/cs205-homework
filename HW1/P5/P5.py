from pyspark import SparkContext
from P5_connected_components import cc_2, get_symmetric_graph
from P5_bfs import BFS

def get_links(link_str):
    """ Takes a string formatted as provided by Wikipedia, i.e.:
    "[page_number]: first_link second_link ... last_link"
    and returns the adjacency list representation given by
    (page number, [first_link, second_link, ...])
    """

    # Splitting at the colon gives us the page and then the links all in one string 
    page, links = link_str.split(': ')

    # Acces the links individuall
    links = [int(link) for link in links.split(' ')]

    # Cast the page to an int for easy indexing
    return (int(page), links)

if __name__ == "__main__":

    # First get a SparkContext object
    sc = SparkContext(appName='shortest_paths', pyFiles=['P5_bfs.py', 'P5_connected_components.py'])
    sc.setLogLevel('WARN')

    # Load in the dataset
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

    # Get the pages with their indices and correct for 1-indexing
    # Our graph will be represented as a list of (page_num, [linked_page_num_1, ...])
    # So we will need to look up the pages by name in this RDD to find out their page number
    # for use in the actual graph
    pages_with_indices = page_names.zipWithIndex().map(lambda (n, id): (id+1, n))

    # Now sort it and cache it for fast lookups
    pages_with_indices = pages_with_indices.sortByKey().cache()

    # Now get our graph, partition it, and cache it
    page_graph = links.map(get_links).partitionBy(256).cache()

    # Get the symmetric subgraph
#    symmetric = get_symmetric_graph(page_graph) 

    # Now find the connected components
#    with open('output.txt', 'w') as out_file:
#        print >> out_file, 'Number of ccs and size of the largest in totally undirected graph:', cc_2(sc, page_graph)
#        print >> out_file, 'Number of ccs and size of the largest in symmetric subgraph:', cc_2(sc, symmetric)

    # Now we need to find the id's corresponding to Kevin Bacon and Harvard
    # Note that because we store our graph as (id, n) we can't do a simple lookup
    # and need to do this somewhat convoluted filter
    Kevin_Bacon = pages_with_indices.filter(lambda (k, v): v == 'Kevin_Bacon').collect()
    assert len(Kevin_Bacon) == 1
    kb_id = Kevin_Bacon[0][0]

    harvard_university = pages_with_indices.filter(lambda (k, v): v == 'Harvard_University').collect()
    assert len(harvard_university) == 1
    hu_id = harvard_university[0][0]

    # Do the searches
    harv_bacon_dist = BFS(page_graph, hu_id, kb_id, sc)
    bacon_harv_dist = BFS(page_graph, kb_id, hu_id, sc)

    # Output the data
    with open('P5.txt','w') as out_file:
        print >> out_file, "Distance from Harvard to Bacon and the path:", harv_bacon_dist
        print >> out_file, "Distance from Bacon to Harvard and the path:", bacon_harv_dist
