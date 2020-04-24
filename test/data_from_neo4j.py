import sys
from py2neo import Graph
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_from_neo4j_database():
    """
    connection to database
    :return: graph
    """
    with open('../neo_config/neoConfig.json') as f:
        # load data within configuration
        try:
            neo_config = json.load(f)
        except:

            sys.exit('Failure to retrieve data...')

    url = neo_config['neodb']['url']
    password = neo_config['neodb']['password']
    graph = Graph(url, password=password)

    return graph


def count_active_relationships_for_user(rel_type, graph):
    """

    :param rel_type:
    :param graph:
    :return: dict with active relationships
    """
    result = graph.run("MATCH (u:User)-[r:" + rel_type + "]->() RETURN u.id as userId, count(r) as relationshipCount",
                       rel_type=rel_type)

    return result.data()


def count_passive_relationships_for_user(rel_type, graph):
    """

    :param rel_type:
    :param graph:
    :return: passive relationships
    """
    result = graph.run("MATCH ()-[r:" + rel_type + "]->(u:User) RETURN u.id as userId, count(r) as relationshipCount",
                       rel_type=rel_type)

    return result.data()


def count_pas_active_relations(rel_type, graph):
    """

    :param rel_type:
    :param graph:
    :return: relations
    hip per user regardless whether is active or passive
    """
    result = graph.run("MATCH ()-[r:" + rel_type + "]-(u:User) RETURN u.id as userId, count(r) as relationshipCount",
                       rel_type=rel_type)

    return result.data()


def total_interactions_per_user(attacks, trades, messages, attacks_dict_p, trades_dict_p, messages_dict_p):
    """
    Convert dicts to dataframes and concatenate by direction of edge
    :param attacks: all dicts
    :param trades:
    :param messages:
    :param attacks_dict_p:
    :param trades_dict_p:
    :param messages_dict_p:
    :return: dicts with for all the users that contain key :user id / value = all the moves which have done(sum all dfs)
    ACTIVE/PASSIVE
    """

    # ACTIVE dfs
    attacks_df = pd.DataFrame.from_records(attacks)
    trades_df = pd.DataFrame.from_records(trades)
    messages_df = pd.DataFrame.from_records(messages)
    # PASSIVE dfs
    attacks_dict_pdf = pd.DataFrame.from_records(attacks_dict_p)
    trades_dict_pdf = pd.DataFrame.from_records(trades_dict_p)
    messages_dict_p_pdf = pd.DataFrame.from_records(messages_dict_p)
    # sum per active user
    active_users = pd.concat([attacks_df, trades_df, messages_df])
    active_sums = active_users.groupby('userId').sum().reset_index()
    # sum per passive user
    passive_users = pd.concat([attacks_dict_pdf, trades_dict_pdf, messages_dict_p_pdf])
    passive_sums = passive_users.groupby('userId').sum().reset_index()

    return active_sums, passive_sums, active_users, passive_users


def total_degrees(total_attacks, total_trades, total_messages):
    """

    :param total_attacks:
    :param total_trades:
    :param total_messages:
    :return: total_in_out degree per user regardless the direction or type of edge
    """

    attacks_df = pd.DataFrame.from_records(total_attacks)
    trades_df = pd.DataFrame.from_records(total_trades)
    messages_df = pd.DataFrame.from_records(total_messages)

    total_df = pd.concat([attacks_df, trades_df, messages_df])

    total_in_out = total_df.groupby('userId').sum().reset_index()
    return total_in_out, attacks_df, trades_df, messages_df


def best_active_users(attacks, trades, messages):
    """

    :param attacks:
    :param trades:
    :param messages:
    :return: active minimum and maximum users in each move
    """
    popular_attacker = max(attacks, key=lambda k: k['relationshipCount'])
    popular_trader = max(trades, key=lambda k: k['relationshipCount'])
    popular_messager = max(messages, key=lambda k: k['relationshipCount'])
    unpopular_attacker = min(attacks, key=lambda k: k['relationshipCount'])
    unpopular_trader = min(trades, key=lambda k: k['relationshipCount'])
    unpopular_messager = min(messages, key=lambda k: k['relationshipCount'])

    return popular_attacker, popular_trader, popular_messager, unpopular_attacker, unpopular_trader, unpopular_messager


def best_passive_users(attacks_dict_p, trades_dict_p, messages_dict_p):

    popular_passive_attacker = max(attacks_dict_p, key=lambda k: k['relationshipCount'])
    popular_passive_trader = max(trades_dict_p, key=lambda k: k['relationshipCount'])
    popular_passive_messager = max(messages_dict_p, key=lambda k: k['relationshipCount'])
    unpopular_passive_attacker = min(attacks_dict_p, key=lambda k: k['relationshipCount'])
    unpopular_passive_trader = min(trades_dict_p, key=lambda k: k['relationshipCount'])
    unpopular_passive_messager = min(messages_dict_p, key=lambda k: k['relationshipCount'])

    return popular_passive_attacker, popular_passive_trader, popular_passive_messager, unpopular_passive_attacker, unpopular_passive_trader, unpopular_passive_messager


def find_min_max_in_total_interactions(active_sums, passive_sums, active_users, passive_users):
    popular_ac_user = max(active_sums, key=lambda k: k['relationshipCount'])
    popular_pa_user = max(passive_sums, key=lambda k: k['relationshipCount'])
    unpopular_ac_user = min(active_sums, key=lambda k: k['relationshipCount'])
    unpopular_pa_user = min(passive_sums, key=lambda k: k['relationshipCount'])
    # ac = active_sums.describe(include='all')
    # print(ac)
    return popular_ac_user, popular_pa_user, unpopular_ac_user, unpopular_pa_user


def average_values(attacks, trades, messages, attacks_dict_p, trades_dict_p, messages_dict_p):
    avg_at = sum(d['relationshipCount'] for d in attacks) / len(attacks)
    avg_tr = sum(d['relationshipCount'] for d in trades) / len(trades)
    avg_mes = sum(d['relationshipCount'] for d in messages) / len(messages)
    avg_at_pas = sum(d['relationshipCount'] for d in attacks_dict_p) / len(attacks_dict_p)
    avg_tr_pas = sum(d['relationshipCount'] for d in trades_dict_p) / len(trades_dict_p)
    avg_mes_pas = sum(d['relationshipCount'] for d in messages_dict_p) / len(messages_dict_p)

    return avg_at, avg_tr, avg_mes, avg_at_pas, avg_tr_pas, avg_mes_pas


def average_from_all_movements(attacks_df, trades_df, messages_df):
    average_at = attacks_df.stack().mean()
    average_tr = trades_df.stack().mean()
    average_mes = messages_df.stack().mean()
    x = 1
    return average_at, average_tr, average_mes


def jac_sim(total_attacks, total_trades, total_messages):
    attacks_for_sim = pd.DataFrame.from_records(total_attacks)
    trades_for_sim = pd.DataFrame.from_records(total_trades)
    messages_for_sim = pd.DataFrame.from_records(total_messages)

    best_in_attacks = attacks_for_sim.nlargest(100, 'relationshipCount')
    best_in_trades = trades_for_sim.nlargest(100, 'relationshipCount')
    best_in_messages = messages_for_sim.nlargest(100, 'relationshipCount')
    best1 = best_in_attacks['userId']
    best2 = best_in_trades['userId']
    best3 = best_in_messages['userId']

    b1 = best1.to_numpy()
    b2 = best2.to_numpy()
    b3 = best3.to_numpy()


    b1 = set(b1)
    b2 = set(b2)
    b3 = set(b3)
    j1 = len(set(b1 & b2)) / len(set(b1 | b2))
    j2 = len(set(b1 & b3)) / len(set(b1 | b3))
    j3 = len(set(b3 & b2)) / len(set(b3 | b2))
    lst = []
    lst.append(j1)
    lst.append(j2)
    lst.append(j3)

    objects = ['att-tra', 'att-mess', 'mess-tra']
    y_pos = np.arange(len(objects))
    lst = set([float(i) for i in lst])
    lst = sorted(lst)
    plt.bar(y_pos, lst)
    plt.title('Metrics for Attacks')
    plt.xticks(y_pos, objects)
    plt.xlabel('metrics')
    plt.ylabel("count")
    plt.show()

    print(j1, j2, j3)
    return j1, j2, j3


def plots(active_sums, passive_sums):
    hist_plot = active_sums['relationshipCount'].hist(bins=50)
    hist_plot.set_title('Active Sums Histogram')
    hist_plot.set_xlabel('Score Value')
    hist_plot.set_ylabel("numbers")
    plt.show()
    hist_plot = passive_sums['relationshipCount'].hist(bins=50)
    hist_plot.set_title('Passive Sums Histogram')
    hist_plot.set_xlabel('Score Value')
    hist_plot.set_ylabel("numbers")
    plt.show()


def plots1(attacks_dict, trades_dict, messages_dict):
    dict_helper = {
        'Attacks Histogram': attacks_dict,
        'Trades Histogram': trades_dict,
        'Messages Histogram': messages_dict
    }
    for hist_title, values in dict_helper.items():
        values_df = pd.DataFrame.from_records(values)
        hist_plot = values_df['relationshipCount'].hist(bins=50)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score Value')
        hist_plot.set_ylabel("numbers")
        plt.show()


def plots2(attacks, trades, messages, attacks_dict_p, trades_dict_p, messages_dict_p):
    attacks_active = pd.DataFrame.from_records(attacks)
    hist_plot = attacks_active['relationshipCount'].hist(bins=50)
    hist_plot.set_title('Active Attacks Histogram')
    hist_plot.set_xlabel('Score Value')
    hist_plot.set_ylabel("numbers")
    plt.show()
    trades_active = pd.DataFrame.from_records(trades)
    hist_plot = trades_active['relationshipCount'].hist(bins=50)
    hist_plot.set_title('Active Trades Histogram')
    hist_plot.set_xlabel('Score Value')
    hist_plot.set_ylabel("numbers")
    plt.show()
    messages_active = pd.DataFrame.from_records(messages)
    hist_plot = messages_active['relationshipCount'].hist(bins=50)
    hist_plot.set_title('Active Messages Histogram')
    hist_plot.set_xlabel('Score Value')
    hist_plot.set_ylabel("numbers")
    plt.show()
    attacks_passive = pd.DataFrame.from_records(attacks_dict_p)
    hist_plot = attacks_passive['relationshipCount'].hist(bins=50)
    hist_plot.set_title('Passive Attacks Histogram')
    hist_plot.set_xlabel('Score Value')
    hist_plot.set_ylabel("numbers")
    plt.show()
    trades_passive = pd.DataFrame.from_records(trades_dict_p)
    hist_plot = trades_passive['relationshipCount'].hist(bins=50)
    hist_plot.set_title('Passive Trades Histogram')
    hist_plot.set_xlabel('Score Value')
    hist_plot.set_ylabel("numbers")
    plt.show()
    messages_passive = pd.DataFrame.from_records(messages_dict_p)
    hist_plot = messages_passive['relationshipCount'].hist(bins=50)
    hist_plot.set_title('Passive Messages Histogram')
    hist_plot.set_xlabel('Score Value')
    hist_plot.set_ylabel("numbers")
    plt.show()


def plots3(popular_attacker, popular_trader, popular_messager, unpopular_attacker, unpopular_trader, unpopular_messager,
           average_at, average_tr, average_mes):
    lst = []
    lst2 = []
    lst3 = []

    a = list(popular_attacker.values())
    b = list(unpopular_attacker.values())

    lst.append(float(average_at))
    lst.append(a[1])
    lst.append(b[1])

    objects = ['min', 'avg', 'max']
    y_pos = np.arange(len(objects))
    lst = set([float(i) for i in lst])
    x = 1
    lst = sorted(lst)
    plt.bar(y_pos, lst)
    plt.title('Metrics for Attacks')
    plt.xticks(y_pos, objects)
    plt.xlabel('metrics')
    plt.ylabel("count")

    plt.show()

    # c= list(popular_trader.values())
    # d = list(unpopular_trader())
    #
    # lst2.append(c[1])
    # lst2.append(d[1])
    # lst2.append((average_tr))
    #
    # e= list(popular_messager.values())
    # f= list(unpopular_messager.values())
    #
    # lst3.append(e)
    # lst3.append(f)
    # lst3.append(average_mes)


def main():
    #graph = read_from_neo4j_database()
    graph = Graph('127.0.0.1', password='leomamao971')
    print("Read from database")
    attacks = count_active_relationships_for_user("ATTACKS", graph)
    trades = count_active_relationships_for_user("TRADES", graph)
    messages = count_active_relationships_for_user("messages", graph)
    attacks_dict_p = count_passive_relationships_for_user("ATTACKS", graph)
    trades_dict_p = count_passive_relationships_for_user("TRADES", graph)
    messages_dict_p = count_passive_relationships_for_user("messages", graph)
    total_attacks = count_pas_active_relations("ATTACKS", graph)
    total_trades = count_pas_active_relations("TRADES", graph)
    total_messages = count_pas_active_relations("messages", graph)
    print('Calculate Active/Passive moves per user')
    active_sums, passive_sums, active_users, passive_users = total_interactions_per_user(attacks, trades, messages,
                                                                                         attacks_dict_p, trades_dict_p,
                                                                                         messages_dict_p)
    x=1
    print('Calculate in/out degree per user')
    total_in_out, attacks_df, trades_df, messages_df = total_degrees(total_attacks, total_trades, total_messages)
    print('average values')
    average_at, average_tr, average_mes = average_from_all_movements(attacks_df, trades_df, messages_df)
    popular_attacker, popular_trader, popular_messager, unpopular_attacker, unpopular_trader, unpopular_messager = best_active_users(attacks, trades, messages)
    # plots(active_sums, passive_sums)
    # # plots for passive users
    # plots1(attacks_df, trades_df, messages_df)
    # # plots for all categories
    # plots2(attacks, trades, messages, attacks_dict_p, trades_dict_p, messages_dict_p)
    #
    #
    # plots3(popular_attacker, popular_trader, popular_messager, unpopular_attacker, unpopular_trader,unpopular_messager, average_at, average_tr, average_mes)
    #
    # popular_passive_attacker, popular_passive_trader, popular_passive_messager, unpopular_passive_attacker, unpopular_passive_trader, unpopular_passive_messager = best_passive_users(attacks_dict_p, trades_dict_p, messages_dict_p)
    # popular_ac_user, popular_pa_user, unpopular_ac_user, unpopular_pa_user = find_min_max_in_total_interactions(active_sums, passive_sums, active_users, passive_users)
    # avg_at, avg_tr, avg_mes, avg_at_pas, avg_tr_pas, avg_mes_pas = average_values(attacks, trades, messages, attacks_dict_p, trades_dict_p, messages_dict_p)
    # ac = find_min_max_in_total_interactions(active_sums, passive_sums, active_users, passive_users)


    jaccard_metric1, jaccard_metric2, jaccard_metric3 = jac_sim(total_attacks, total_trades, total_messages)


if __name__ == '__main__':
    main()
