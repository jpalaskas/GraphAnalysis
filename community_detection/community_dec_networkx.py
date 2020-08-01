import community
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import glob
import ujson
from collections import defaultdict
import itertools




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
    # find partition
    partition = community.best_partition(unGraph)

    # find modularity
    modularity = community.modularity(partition, unGraph)
    print(modularity)

    # write results to txt file
    louvain_results = defaultdict(list)
    for node, comm in sorted(partition.items()):
        louvain_results[comm].append(node)
    with open('louvain_results.txt', 'a') as f:
        f.write(ujson.dumps(louvain_results))
        f.write('modularity= ' + str(modularity))

    # visualization
    values = [partition.get(node) for node in unGraph.nodes()]
    nx.draw_spring(unGraph, cmap=plt.get_cmap('jet'), node_color=values,
                   node_size=20, with_labels=False)
    plt.show()


def label_propagation(unGraph):
    # find communities
    lbcomms = nx.algorithms.community.label_propagation.label_propagation_communities(unGraph)

    # write results to txt file
    ress = {}
    res = list(lbcomms)
    count = 0
    for item in res:
        ress[str(count)] = list(item)
        count += 1

    # find modularity
    modularity = community.modularity(ress, unGraph)
    print(modularity)

    with open('label_propagation_results.txt', 'a') as f:
        f.write(ujson.dumps(ress))
        f.write('modulariry= ' + str(modularity))

    # visualization (not working?)
    values = [ress.get(node) for node in unGraph.nodes()]
    nx.draw_spring(unGraph, cmap=plt.get_cmap('jet'), node_color=values,
                   node_size=20, with_labels=False)
    plt.show()


def async_label_propagation():
    pass


def girvan_newman(unGraph):
    gncomms = nx.algorithms.community.girvan_newman(unGraph)
    k = 500
    for communities in itertools.islice(gncomms, k):
        print(tuple(sorted(c) for c in communities))


def kernighan_lin_bipartition():
    pass


def k_clique():
    pass


def greedy_modularity():
    pass


def async_fluid_communities():
    pass


def main():
    g_directed, g_undirected, all_dfs, dir_labels, undir_labels = createGraphs()

    # working
    louvain(g_undirected)

    #working (check visualization)
    label_propagation(g_undirected)

    # girvan_newman(g_undirected)


if __name__ == '__main__':
    main()