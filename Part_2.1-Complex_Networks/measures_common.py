import networkx as nx
import statistics
import warnings
warnings.filterwarnings('ignore')

def get_measures_from_graph(G):

    # Centrality
    betweenness = nx.betweenness_centrality(G) 
    closeness = nx.closeness_centrality(G)

    try: 
        eigenvector = nx.eigenvector_centrality_numpy(G, max_iter=1000, tol=1e-05)
    except: 
        print('Eigenvector centrality failed')
        eigenvector = 0

    katz = nx.katz_centrality_numpy(G)
    pagerank = nx.pagerank(G)
    hubs,authorities = nx.hits(G, max_iter=1000)

    # Clustering
    clustering = nx.clustering(G)
    average_clustering = nx.average_clustering(G)
    correlation = nx.degree_pearson_correlation_coefficient(G)
    transitivity = nx.transitivity(G)

    #Other
    density = nx.density(G)

    measures = {
    'betweenness': betweenness,
    'closeness': closeness,
    'eigenvector': eigenvector,
    'katz': katz,
    'pagerank': pagerank,
    'hubs': hubs,
    'authorities': authorities,
    'clustering': clustering,
    'average_clustering': average_clustering,
    'correlation': correlation,
    'transitivity': transitivity,
    'density': density
    }

    return measures


def get_measures_from_text(text):
    
    G = nx.DiGraph()
    text = text.split()

    G.add_nodes_from(text)

    for i in range(len(text) - 1):
        G.add_edge(text[i], text[i + 1])

    measures = get_measures_from_graph(G)

    for key, value in list(measures.items()):
        if  isinstance(value, dict):
            measures[key] = statistics.mean(value.values())
    
    return measures
