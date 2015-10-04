from pyspark import SparkContext
from P4_bfs import BFS
import time

def split_up_input(line):
    """ Takes in a string, assumed to be of the form '"character", "issue"'
    and returns a tuple of (character, issue)"""

    split_line = line.split('"')
    character = split_line[1]
    issue = split_line[3]

    return (character, issue)

def get_all_individual_pairs(issue_list_tup):
    """ Takes a list of characters, all of which are assumed to appear in 
    the same issue, and returns all combinations of (char1, char2). This is
    sort of like a really poorly thought out adjacency matrix. For use in conjunction with flatMap
    and to be post-processed into an adjacency list representation.
    """

    list_of_chars = issue_list_tup[1]

    edges = []
    for c1 in list_of_chars:
        for c2 in list_of_chars:
            if (c1 != c2):              # avoid self-links 
                edges.append((c1, c2))
                edges.append((c2, c1))

    return tuple(edges)                 # flatMap expects a tuple

if __name__ == "__main__":

    start_time = time.time()

    # First get a SparkContext object
    sc = SparkContext('local', 'BFS')
    sc.setLogLevel('WARN')

    # Now load in the data
    data = sc.textFile('source.csv', 16)

    # Get rid of the stupid quotes
    chars_and_issues = data.map(split_up_input)

    # Flip it around for a groupByKey 
    issues_and_chars = chars_and_issues.map(lambda (x, y): (y, x))

    # Now we have the issues as keys - all things with the same issue should be connected
    issues_and_chars = issues_and_chars.groupByKey()

    # Now we access every single edge that appears in the graph
    all_edges = issues_and_chars.flatMap(get_all_individual_pairs)

    # Now each key is a character, and it appears many times -
    # once for each edge. Consolidate all of these into an adjacency
    # list representation of the graph.
    # We are on 4 cores so we give it 16 partitions. We will try to preserve this partitioning
    # as we progress. We did not worry about the partitioning scheme of the previous RDD's
    # because they were so transient.
    marvel_graph = all_edges.groupByKey().partitionBy(16)

    # Clean up any duplicates that may appear
    # We preserve our partitioning scheme because we do not alter the key
    marvel_graph = marvel_graph.map(lambda (x, y): (x, list(set(y))), preservesPartitioning = True)
    
    # Cache our graph to speed up the searches
    marvel_graph = marvel_graph.cache()

    # Log the time to set up the graph
    # Note: may be inaccurate due to laziness
    print 'Setup time:', (time.time() - start_time)

    # Now run the first BFS
    ca_graph = BFS(marvel_graph,'CAPTAIN AMERICA', sc)

    # Filter out all nodes who never got their distances updated
    num_touched_ca = ca_graph.filter(lambda (x, y): True if y[0] < 10**8 else False).count()

    # Now run the second BFS
    mtm_graph = BFS(marvel_graph,'MISS THING/MARY', sc)

    # Filter out all nodes who never got their distances updated
    num_touched_mtm = mtm_graph.filter(lambda (x, y): True if y[0] < 10**8 else False).count()

    # Now run the last BFS
    o_graph = BFS(marvel_graph,'ORWELL', sc)

    # Filter out all graphs who never got their distances updated
    num_touched_o = o_graph.filter(lambda (x, y): True if y[0] < 10**8 else False).count()


    with open('P4_output.txt', 'w') as output_file:
        output_file.write('Num touched for orwell: ' + str(num_touched_o))
        output_file.write('Num touched for captain america: ' + str(num_touched_ca))
        output_file.write('Num touched for miss thing: ' + str(num_touched_mtm))

    print 'Total time:', (time.time() - start_time)
