import community 
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import glob
import ujson
from collections import defaultdict 
import itertools
import sys
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering, spectral_clustering

def createGraphs():
    files = glob.glob("*.csv")

    all_dfs = pd.DataFrame(columns=['Timestamp', 'id1', 'id2', 'label'])

    for filename in files:
        print('Currently using file - ', filename)

        df = pd.read_csv(filename, header=None)
        df.columns = ['Timestamp', 'id1', 'id2']

        if 'attack' in filename:
            rel_type = 'attacks'
        elif 'trade' in filename:
            rel_type = 'trades'
        else:
            rel_type = 'messages'

        df['label'] = rel_type

        all_dfs = pd.concat([all_dfs, df])
    
    g_directed = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
        create_using=nx.MultiDiGraph(name='Travian_Graph'))

    g_undirected = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
        create_using=nx.MultiGraph(name='Travian_Graph'))

    dir_labels = {e: g_directed.edges[e]['label'] for e in g_directed.edges}
    
    undir_labels = {e: g_undirected.edges[e]['label'] for e in g_undirected.edges}
    
    return g_directed, g_undirected, all_dfs, dir_labels, undir_labels


def louvain(unGraph):
    #find partition
    partition = community.best_partition(unGraph)
    print(partition)
    #find modularity
    modularity = community.modularity(partition, unGraph)
    print(modularity)
    

    #write results to txt file
    louvain_results = defaultdict(list)
    for node, comm in sorted(partition.items()):
        louvain_results[comm].append(node)

    for item in louvain_results:
        print('For community=' + str(item) + ':')
        # print(louvain_results[item])
        subG = unGraph.subgraph(louvain_results[item])
        print('NumOfEdges '+str(len(subG.edges())))
        print('NumOfNodes '+str(len(subG.nodes())))
        print('Density of community ='+str( len(subG.edges()) / len(subG.nodes())))
        print('Max node '+str(max(dict(subG.degree()).items(), key = lambda x : x[1])))
        print('Min node '+str(min(dict(subG.degree()).items(), key = lambda x : x[1])))
    
    with open('louvain_results.txt', 'a') as f:
        f.write(ujson.dumps(louvain_results))
        f.write('modularity= ' + str(modularity))

    #visualization
    values = [partition.get(node) for node in unGraph.nodes()]
    nx.draw_spring(unGraph, cmap = plt.get_cmap('jet'), node_color = values,
        node_size=20, with_labels=False)
    plt.show()

def label_propagation(unGraph):
    #find communities
    lbcomms = nx.algorithms.community.label_propagation.label_propagation_communities(unGraph)

    #write results to txt file
    ress = {}
    res = list(lbcomms)
    count = 0
    for item in res:
        ress[str(count)] = list(item)
        count += 1
    
    #find modularity
    modularity = community.modularity(ress, unGraph)
    print(modularity)

    

    with open('label_propagation_results2.txt', 'a') as f:
        f.write(ujson.dumps(ress))
        f.write('modulariry= ' + str(modularity))

    #visualization (not working?)
    values = [ress.get(node) for node in unGraph.nodes()]
    nx.draw_spring(unGraph, cmap = plt.get_cmap('jet'), node_color = values,
        node_size=20, with_labels=False)
    plt.show()

def edge_to_remove(unGraph):
    d = nx.edge_betweenness_centrality(unGraph)
    # list_of_tuples = d.items()
    # list_of_tuples.sort(key= lambda x:x[1], reverse=True)
    list_of_tuples = sorted(d.items(), key= lambda x:x[1], reverse=True)
    return list_of_tuples[0][0]

def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def girvan_newman(unGraph):
    gncomms = list(nx.algorithms.community.centrality.girvan_newman(unGraph))
    print(gncomms)

    # subGraphs = connected_component_subgraphs(unGraph)
    # length = len(list(subGraphs))
    # print('The number of connected components are', length)
    # while(length!=2):
    #     unGraph.remove_edge(*edge_to_remove(unGraph))
    #     subGraphs = connected_component_subgraphs(unGraph)
    #     length = len(list(subGraphs))
    #     print('The number of connected components are', length)
    # count = 0
    # print(list(subGraphs))
    # for i in list(subGraphs):
    #     print('Community #' + str(i+1) + '-----------------')
    #     print(i.nodes())
    #     print('************************************')
    #     count += 1

def gn(G):

    if len(G.nodes()) == 1:
        return [G.nodes()]

    def find_best_edge(G0):
        """
        Networkx implementation of edge_betweenness
        returns a dictionary. Make this into a list,
        sort it and return the edge with highest betweenness.
        """
        eb = nx.edge_betweenness_centrality(G0)
        eb_il = eb.items()
        eb_il_s = sorted(eb_il, key=lambda x: x[1], reverse=True)
        return eb_il_s[0][0]

    components = connected_component_subgraphs(G)

    while len(list(components)) == 1:
        G.remove_edge(*find_best_edge(G))
        components = connected_component_subgraphs(G)

    result = [c.nodes() for c in components]

    for c in components:
        print(c.nodes())
        result.extend(gn(c))

    return (result)

def k_clique(unGraph):
    # fcq = list(nx.algorithms.clique.find_cliques(unGraph))
    # for item in (fcq):
    #     print(list(item))
    # print('found cliques')
    # print('*'*20)   

    comms = list(nx.algorithms.community.k_clique_communities(unGraph, 4))
    count = 0
    # for item in (comms):
    #     print(list(item))
    #     count += len(list(item))
    # print(count)
    ress = dict()
    for item in comms:
        ress[str(count)] = list(item)
        count += 1

    modularity = community.modularity(ress, unGraph)
    print(modularity)

    with open('kclique_results.txt', 'a') as f:
        f.write(ujson.dumps(ress))
        f.write('modulariry= ' + str(modularity))

def clauset_newman_moore(unGraph):
    c = list(nx.algorithms.community.greedy_modularity_communities(unGraph))
    for item in (c):
        print(list(item))

def sc(unGraph):
    np.set_printoptions(threshold=sys.maxsize)
    A = nx.convert_matrix.to_numpy_matrix(unGraph)

    nodes = list(unGraph.nodes())
    # print(nodes)
    # print(len(nodes))
    nodes.sort()
    # print(nodes)
    clusters = SpectralClustering(affinity = 'precomputed', 
        assign_labels="kmeans",random_state=0,n_clusters=200).fit_predict(A)
    # print('modularity = '+ str(community.modularity(clusters, unGraph)))
    print(clusters)
    print(len(clusters))
    plt.scatter(nodes, clusters, c=clusters, s=50, cmap='viridis')
    plt.show()

    # comms = set(clusters)
    print(set(clusters))

    results = defaultdict(list)
    for node, cluster in zip(nodes, clusters):
        results[str(cluster)].append(node) 

    with open('sc_res_test200.txt','a') as f:
        f.write(ujson.dumps(results))

    nodes.sort(reverse=True)
    
def main():
    g_directed, g_undirected, all_dfs, dir_labels, undir_labels = createGraphs()

    #working
    louvain(g_undirected)

    #working (check visualization)
    # label_propagation(g_undirected)
    # girvan_newman(g_undirected)
    # gn(nx.karate_club_graph())

    # k_clique(g_undirected)

    # clauset_newman_moore(g_undirected)

    # spectral_clustering(g_undirected)
    # sc(g_undirected)
    #working

if __name__ == '__main__':
    main()