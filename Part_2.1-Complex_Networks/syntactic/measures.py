import networkx as nx
from tqdm.notebook import tqdm
import os
import statistics
import numpy as np
import math

def get_measures_from_graph(G):

    # Centrality
    betweenness = nx.betweenness_centrality(G) 
    closeness = nx.closeness_centrality(G)

    try: 
        eigenvector = nx.eigenvector_centrality_numpy(G, max_iter=1000, tol=1e-05)
    except: 
        eigenvector = 0

    katz = nx.katz_centrality_numpy(G)
    pagerank = nx.pagerank(G)
    try:
        hubs,authorities = nx.hits(G, max_iter=1000)
    except:
        hubs = 0
        authorities = 0

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
    'density': density
    }

    return measures

def generate_graph_udpipe(file_path):

    G = nx.DiGraph()
    G.add_node(0, word="<root>", postag="NONE", deprel= "none")
    edges = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                tuples = line.split()
                G.add_node(int(tuples[0]), word=tuples[1], postag=tuples[3], deprel=tuples[4])
                edges.append((int(tuples[2]), int(tuples[0])))
    
    G.add_edges_from(edges)
    return G




def get_measures_from_graph_word(G):

    # Centrality
    betweenness = nx.betweenness_centrality(G) 
    closeness = nx.closeness_centrality(G)

    try: 
        eigenvector = nx.eigenvector_centrality_numpy(G, max_iter=1000, tol=1e-05)
    except: 
        eigenvector = 0

    katz = nx.katz_centrality_numpy(G)
    pagerank = nx.pagerank(G)
    try:
        hubs,authorities = nx.hits(G, max_iter=1000)
    except:
        hubs = 0
        authorities = 0

    # Clustering

    try:
        clustering = nx.clustering(G)
    except:
        clustering = 0

    average_clustering = nx.average_clustering(G)

    try:
        correlation = nx.degree_pearson_correlation_coefficient(G)
    except:
        correlation = 0

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



def generate_graph_udpipe_word(file_path):

    G = nx.DiGraph()
    G.add_node("<root>", postag="NONE", deprel= "none")
    node_to_word = {0: "<root>"}
    edges_num = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                tuples = line.split()
                G.add_node(tuples[1], postag=tuples[3], deprel=tuples[4])
                node_to_word[int(tuples[0])] = tuples[1]
                edges_num.append((int(tuples[2]), int(tuples[0])))
    
    for edge in edges_num:
        G.add_edge(node_to_word[edge[0]], node_to_word[edge[1]])
    
    return G