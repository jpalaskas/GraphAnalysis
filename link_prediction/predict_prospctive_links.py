import glob

import networkx as nx
import pandas as pd
from tqdm import tqdm


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

        df['type'] = rel_type
        df['weight'] = 1
        all_dfs = pd.concat([all_dfs, df])

    graph = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
                                    create_using=nx.MultiDiGraph(name='Travian_Graph'))
    g_undirected = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
                                           create_using=nx.Graph(name='Travian_Graph'))

    labels = {e: graph.edges[e]['type'] for e in graph.edges}
    return graph, all_dfs, labels, g_undirected


def aggregated_dataset(all_dfs, g_undirected):
    aggregated_df = all_dfs.sort_values(by='Timestamp', ascending=True)
    aggregated_df = all_dfs.groupby(['id1', 'id2', 'type'], as_index=False)['weight'].sum()
    aggregated_df = aggregated_df.set_index(['id1', 'id2'])

    aggregated_df['preferential attachment'] = [i[2] for i in
                                                nx.preferential_attachment(g_undirected, aggregated_df.index)]
    aggregated_df['Common Neighbors'] = aggregated_df.index.map(
        lambda id: len(list(nx.common_neighbors(g_undirected, id[0], id[1]))))
    aggregated_df['label'] = 1
    aggregated_df.to_pickle("./dummy.pkl")
    return aggregated_df


def calc_weeights_without_aggregate(all_dfs):
    pairs = {}
    for index, row in all_dfs.iterrows():
        if (row['id1'], row['id2']) not in pairs:
            pairs[(row['id1'], row['id2'])] = 0
        pairs[(row['id1'], row['id2'])] += 1  # it could be row['weight']

    for index, row in all_dfs.iterrows():
        if (row['id1'], row['id2']) in pairs:
            all_dfs.at[index, 'weight'] = pairs[(row['id1'], row['id2'])]
    all_dfs.to_pickle("./aggregated_df.pkl")
    # df = pd.read_pickle("./aggregated_df.pkl")
    return all_dfs


def pad(df, calendar):
    # create unique id
    sales = df.copy()
    # sales = df[df['date_num']<20190700]
    sales['id'] = sales['item_id'].astype(str) + '_' + sales['store_id'].astype(str)
    # #create new multi index
    pad_index = pd.MultiIndex.from_product([sales['id'].unique(), calendar['date_num'].unique()],
                                           names=['id', 'date_num'])
    # #padding
    padded = sales.set_index(['id', 'date_num']).reindex(pad_index).reset_index()
    # #pivot
    padded = padded.pivot(index='date_num', columns='id', values='sales')

    # remove unnecessary values and filling NaNs with 0
    for i in tqdm(padded.columns):
        first_val = padded[i].first_valid_index()
        last_val = padded[i].last_valid_index()
        padded.loc[first_val:last_val, i].fillna(0, inplace=True)

    padded = padded.stack().reset_index()
    padded.rename(columns={padded.columns[-1]: 'sales'}, inplace=True)
    sales.drop(columns=sales.columns[sales.columns.isin(
        ['sales', 'prices', 'date_num', 't_dw', 't_dm', 't_dy', 't_wm', 't_wy', 't_my', 't_y', 't_w_end'])],
               inplace=True)
    # test
    sales.drop_duplicates(inplace=True)
    sales = sales.merge(padded, on='id', how='left')

    return sales[sales.columns[~sales.columns.isin(['id'])]]


def generate_timestamps():
    df = pd.read_pickle("./aggregated_df.pkl")


def centrality_measures(g_undirected, graph, all_dfs):
    bet_cen = nx.betweenness_centrality(g_undirected)
    close_cen = nx.closeness_centrality(graph)

    all_dfs['betweeness_centrality'] = all_dfs['id1'].map(bet_cen)
    all_dfs['betweeness_centrality fot id2'] = all_dfs['id2'].map(bet_cen)
    all_dfs['closeness_centrality'] = all_dfs['id1'].map(close_cen)
    all_dfs['closeness_centrality for id2'] = all_dfs['id2'].map(close_cen)
    return all_dfs


def map_predictions_to_df(predictions, row):
    if (row['id1'], row['id2']) in predictions:
        return predictions[(row['id1'], row['id2'])]
    else:
        return predictions[(row['id2'], row['id1'])]


def main():
    graph, all_dfs, labels, g_undirected = read_csv_files()

    # all_dfs.to_pickle("./aggregated_dflalala.pkl")


    lst = []
    lst2 = []
    # predictions1 = nx.preferential_attachment(g_undirected, g_undirected.edges())
    #
    # [lst.append((u, v, p)) for u, v, p in predictions1]
    # predictions1 = {(k, v): n for k, v, n in lst}
    #
    # all_dfs['preferential'] = all_dfs.apply(lambda x: map_predictions_to_df(predictions1, x), axis=1)
    #
    # predictions3 = nx.resource_allocation_index(g_undirected, g_undirected.edges())
    #
    # try:
    #     [lst2.append((u, v, p)) for u, v, p in predictions3]
    #     predictions3 = {(k, v): n for k, v, n in lst2}
    #
    #     all_dfs['Resource_allocation'] = all_dfs.apply(lambda x: map_predictions_to_df(predictions3, x), axis=1)
    #
    # except ZeroDivisionError:
    #     print("ZeroDivisionError: float division by zero")


    # all_dfs['Common Neighbors'] = all_dfs['id1'].map(lambda city: (list(nx.common_neighbors(graph, city[0], city[1]))))


    predictions1 = nx.common_neighbors(g_undirected, g_undirected.edges())




    [lst2.append((u, v, p)) for u, v, p in predictions2]
    predictions1 = {(k, v): n for k, v, n in lst2}

    all_dfs['Common_neighboors'] = all_dfs.apply(lambda x: map_predictions_to_df(predictions2, x), axis=1)


    adamic_adar = list(all_dfs=nx.adamic_adar_index(g_undirected, g_undirected.edges()))
    all_dfs = pd.DataFrame(index=[(x[0], x[1]) for x in adamic_adar])
    all_dfs['adamic_adar'] = [x[2] for x in adamic_adar]


    all_dfs['Common Neighbors'] = all_dfs.index.map(
        lambda id: len(list(nx.common_neighbors(g_undirected, id[0], id[1]))))
    df = pd.read_pickle("./aggregated_df.pkl")
    # dict = g_undirected.edge_betweenness_centrality()

    # values_df = pd.DataFrame(dict.items(), columns=['id', 'Score'])

    preferential_attachment = list(nx.preferential_attachment(g_undirected))
    all_dfs = pd.DataFrame(index=[(x[0], x[1]) for x in preferential_attachment])
    all_dfs['preferential_attachment'] = [x[2] for x in preferential_attachment]

    all_dfs['Common Neighbors'] = all_dfs.index.map(
        lambda id: len(list(nx.common_neighbors(g_undirected, id[0], id[1]))))


if __name__ == '__main__':
    main()

    # graph.edges(data=True)
    # df = pd.DataFrame(index=graph.edges()).reset_index()
    # df['weight'] = pd.Series(nx.get_edge_attributes(graph, 'weight'))

    # all_dfs = all_dfs.sort_values(by='Timestamp', ascending=True)
    # all_dfs = all_dfs.groupby(['id1', 'id2', 'label'], as_index=False)['weight'].sum()
    # all_dfs = all_dfs.set_index(['id1', 'id2'])

    # all_dfs = all_dfs.groupby(['Timestamp', 'label'], as_index=False)['weight'].sum()
    # all_dfs = all_dfs.index.names['ids']
    # all_dfs.index.names = ['Ticker', all_dfs.index.names[1]]
    # all_dfs.set_index('name', inplace=True)
    # all_dfs = all_dfs.sort_values(by='Timestamp', ascending=True).reset_index(['id1', 'id2'])
    # all_dfs = all_dfs.groupby(['label'], as_index=False)['weight'].sum().reset_index()
    # all_dfs = all_dfs.set_index(['id1', 'id2'], inplace=True, append=True, drop=False)
    # machine_learning(graph, all_dfs)

    # all_dfs['preferential attachment'] = [i[2] for i in nx.preferential_attachment(g_undirected, all_dfs.index)]
    # all_dfs = all_dfs.groupby(['label', 'preferential attachment'], as_index=False)['weight'].sum()
    # x=1
