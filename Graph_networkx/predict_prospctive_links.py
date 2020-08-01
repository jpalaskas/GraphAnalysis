import glob
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


def read_csv_files():
    """
    Read data and create MultiDiGraph.Each Node has an id and All edges have 2 attributes.
    The first is Timestamp and the second is the type of edge (Attacks, Trades, Messages)
    :return: G, all_dfs, labels
    """
    file_names = glob.glob("../data_users_moves/*.csv")

    all_dfs = pd.DataFrame(columns=['Timestamp', 'id1', 'id2', 'label'])

    for file in file_names:
        print(str(file))
        df = pd.read_csv(file, header=None)
        df.columns = ['Timestamp', 'id1', 'id2']
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        # df['date'] = [d.date() for d in df['Timestamp']]
        # df['time'] = [d.time() for d in df['Timestamp']]
        if 'attack' in file:
            rel_type = 'attacks'
        elif 'trade' in file:
            rel_type = 'trades'
        else:
            rel_type = 'messages'

        df['label'] = rel_type
        df['weight'] = 1
        all_dfs = pd.concat([all_dfs, df])

    graph = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
                                    create_using=nx.MultiDiGraph(name='Travian_Graph'))
    g_undirected = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
                                           create_using=nx.Graph(name='Travian_Graph'))

    labels = {e: graph.edges[e]['label'] for e in graph.edges}
    return graph, all_dfs, labels, g_undirected

#
# def machine_learning(graph, all_dfs):
#     # def salary_predictions():
#     #     def is_management(node):
#     #         managementSalary = node[1]['ManagementSalary']
#     #         if managementSalary == 0:
#     #             return 0
#     #         elif managementSalary == 1:
#     #             return 1
#     #         else:
#     #             return None
#
#     df = pd.DataFrame(index=graph.nodes())
#     df['clustering'] = pd.Series(nx.clustering(graph))
#     df['degree'] = pd.Series(graph.degree())
#     df['degree_centrality'] = pd.Series(nx.degree_centrality(graph))
#     df['closeness'] = pd.Series(nx.closeness_centrality(graph, normalized=True))
#     df['betweeness'] = pd.Series(nx.betweenness_centrality(graph, normalized=True))
#     df['pr'] = pd.Series(nx.pagerank(graph))
#     #df['is_management'] = pd.Series([is_management(node) for node in G.nodes(data=True)])
#     df_train = df[~pd.isnull(df['is_management'])]
#     df_test = df[pd.isnull(df['is_management'])]
#     features = ['clustering', 'degree', 'degree_centrality', 'closeness', 'betweeness', 'pr']
#     X_train = df_train[features]
#     Y_train = df_train['is_management']
#     X_test = df_test[features]
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     clf = MLPClassifier(hidden_layer_sizes=[10, 5], alpha=5,
#                         random_state=0, solver='lbfgs', verbose=0)
#     clf.fit(X_train_scaled, Y_train)
#     test_proba = clf.predict_proba(X_test_scaled)[:, 1]
#     return pd.Series(test_proba, X_test.index)


def main():
    graph, all_dfs, labels, g_undirected  = read_csv_files()
    # graph.edges(data=True)
    # df = pd.DataFrame(index=graph.edges()).reset_index()
    # df['weight'] = pd.Series(nx.get_edge_attributes(graph, 'weight'))

    # all_dfs = all_dfs.sort_values(by='Timestamp', ascending=True)
    # all_dfs = all_dfs.groupby(['id1', 'id2', 'label'], as_index=False)['weight'].sum()
    # all_dfs = all_dfs.set_index(['id1', 'id2'])
    all_dfs = all_dfs.sort_values(by='Timestamp', ascending=True).reset_index(['id1', 'id2'])
    # all_dfs = all_dfs.groupby(['id1', 'id2', 'label'], as_index=False)['weight'].sum()
    # all_dfs = all_dfs.set_index(['id1', 'id2'], inplace=True, append=True, drop=False)
    # machine_learning(graph, all_dfs)
    all_dfs['preferential attachment'] = [i[2] for i in nx.preferential_attachment(g_undirected, all_dfs.index)]
    all_dfs = all_dfs.groupby(['label', 'preferential attachment'], as_index=False)['weight'].sum()
    x=1


if __name__ == '__main__':
    main()
