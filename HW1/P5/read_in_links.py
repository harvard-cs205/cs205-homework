links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')


def parse_links(x):
    """
    Takes arg of form "source: n1 n2 n3 n4 n5" and returns
    (source, [n1, n2, n3, n4, n5])
    """
    source, neighbors = x.split(': ')
    source = int(source)
    neighbors = [int(i) for i in neighbors.split(' ')]

    return (source, neighbors)

links = links.map(parse_links)
