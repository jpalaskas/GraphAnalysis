from sklearn.metrics import jaccard_score
from py2neo import Graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def count_active_relationships_for_user(rel_type, graph):
    """

    :param rel_type:
    :param graph:
    :return: dict with active relationships
    """
    result = graph.run("MATCH (u:User)-[r:" + rel_type + "]->() RETURN u.id as userId, count(r) as relationshipCount",
                       rel_type=rel_type).to_data_frame()

    return result


def count_passive_relationships_for_user(rel_type, graph):
    """

    :param rel_type:
    :param graph:
    :return: passive relationships
    """
    result = graph.run("MATCH ()-[r:" + rel_type + "]->(u:User) RETURN u.id as userId, count(r) as relationshipCount", rel_type=rel_type).to_data_frame()


    return result


def count_pas_active_relations(rel_type, graph):
    """

    :param rel_type:
    :param graph:
    :return: relationship per user regardless whether is active or passive
    """
    X = 1
    result = graph.run("MATCH ()-[r:" + rel_type + "]-(u:User) RETURN u.id as userId, count(r) as relationshipCount",
                       rel_type=rel_type).to_data_frame()

    return result


def total_interactions_per_user(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df):
    # sum per active user
    active_users = pd.concat([attacks, trades, messages])
    active_sums = active_users.groupby('userId').sum().reset_index()
    # sum per passive user
    passive_users = pd.concat([attacks_p_df, trades_p_df, messages_p_df])
    passive_sums = passive_users.groupby('userId').sum().reset_index()

    return active_sums, passive_sums


def max_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df):
    max_attacks = attacks[attacks.relationshipCount == attacks.relationshipCount.max()]
    max_trades = trades[trades.relationshipCount == trades.relationshipCount.max()]
    max_messages = messages[messages.relationshipCount == messages.relationshipCount.max()]
    max_attacks_p_df = attacks_p_df[attacks_p_df.relationshipCount == attacks_p_df.relationshipCount.max()]
    max_trades_p_df = trades_p_df[trades_p_df.relationshipCount == trades_p_df.relationshipCount.max()]
    max_messages_p_df = messages_p_df[messages_p_df.relationshipCount == messages_p_df.relationshipCount.max()]

    return max_attacks, max_trades, max_messages, max_attacks_p_df, max_trades_p_df, max_messages_p_df


def min_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df):
    min_attacks = attacks[attacks.relationshipCount == attacks.relationshipCount.min]
    min_trades = trades[trades.relationshipCount == trades.relationshipCount.min]
    min_messages = messages[messages.relationshipCount == messages.relationshipCount.min]
    min_attacks_p_df = attacks_p_df[attacks_p_df.relationshipCount == attacks_p_df.relationshipCount.min]
    min_trades_p_df = trades_p_df[trades_p_df.relationshipCount == trades_p_df.relationshipCount.min]
    min_messages_p_df = messages_p_df[messages_p_df.relationshipCount == messages_p_df.relationshipCount.min]

    return min_attacks, min_trades, min_messages, min_attacks_p_df, min_trades_p_df, min_messages_p_df


def average_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df):
    average_at = attacks.stack().mean()
    average_tr = trades.stack().mean()
    average_mes = messages.stack().mean()
    average_pas_at = attacks_p_df.stack().mean()
    average_pas_tr = trades_p_df.stack().mean()
    average_pas_mes = messages_p_df.stack().mean()

    return average_at, average_tr, average_mes, average_pas_at, average_pas_tr, average_pas_mes


def act_pass_metrics(active_sums, passive_sums):
    max_sum = active_sums[active_sums.relationshipCount == active_sums.relationshipCount.max()]
    min_sum= active_sums[active_sums.relationshipCount == active_sums.relationshipCount.min]
    average_sum = active_sums.stack().mean()
    max_sum_passive = passive_sums[passive_sums.relationshipCount == passive_sums.relationshipCount.max()]
    min_sum_passive = passive_sums[passive_sums.relationshipCount == passive_sums.relationshipCount.min]
    average_sum_passive = active_sums.stack().mean()

    return max_sum, min_sum, average_sum, max_sum_passive, min_sum_passive, average_sum_passive


def in_out_degree(total_attacks, total_trades, total_messages):
    max_attacks = total_attacks[total_attacks.relationshipCount == total_attacks.relationshipCount.max()]
    min_attacks = total_attacks[total_attacks.relationshipCount == total_attacks.relationshipCount.min]
    average_attacks = total_attacks.stack().mean()
    max_trades = total_trades[total_trades.relationshipCount == total_trades.relationshipCount.max()]
    min_trades = total_trades[total_trades.relationshipCount == total_trades.relationshipCount.min]
    average_trades = total_trades.stack().mean()
    max_messages = total_messages[total_messages.relationshipCount == total_messages.relationshipCount.max()]
    min_messages = total_messages[total_messages.relationshipCount == total_messages.relationshipCount.min]
    average_messages = total_messages.stack().mean()

    return max_attacks, min_attacks, average_attacks, max_trades, min_trades, average_trades, max_messages, min_messages, average_messages


def plots(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df, active_sums, passive_sums, total_attacks, total_trades, total_messages):
    dict_helper = {
        'Active Attacks Histogram': attacks,
        'Active Trades Histogram': trades,
        'Active Histogram': messages,
        'Passive Messages Histogram': attacks_p_df,
        'Passive Messages Histogram': trades_p_df,
        'Passive Messages Histogram': messages_p_df,
        'Active Sums Histogram': active_sums,
        'Passive sums Histogram': passive_sums,
        'In-out attacks per user': total_attacks,
        'In-out  per user': total_trades,
        'In-out messages per user': total_messages
    }

    for hist_title, values in dict_helper.items():
        hist_plot = values['relationshipCount'].hist(bins=1000)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score Value')
        hist_plot.set_ylabel("numbers")
        plt.show()


def plots_for_min_max_avg(max_attacks, max_trades, max_messages, max_attacks_p_df, max_trades_p_df, max_messages_p_df,
                          min_attacks, min_trades, min_messages, min_attacks_p_df, min_trades_p_df, min_messages_p_df,
                          average_at, average_tr, average_mes, average_pas_at, average_pas_tr, average_pas_mes):
    attacks_active = []
    attacks_active.append(max_attacks['relationshipCount'].values)
    attacks_active.append(min_attacks['relationshipCount'].values)
    attacks_active.append(average_at)

    attacks_passive = []
    attacks_passive.append(max_attacks_p_df['relationshipCount'].values)
    attacks_passive.append(min_attacks_p_df['relationshipCount'].values)
    attacks_passive.append(average_pas_at)

    trades_active = []
    trades_active.append(max_trades['relationshipCount'].values)
    trades_active.append(min_trades['relationshipCount'].values)
    trades_active.append(average_tr)

    trades_passive = []
    trades_passive.append(max_trades_p_df['relationshipCount'].values)
    trades_passive.append(min_trades_p_df['relationshipCount'].values)
    trades_passive.append(average_pas_tr)

    messages_active = []
    messages_active.append(max_messages['relationshipCount'].values)
    messages_active.append(min_messages['relationshipCount'].values)
    messages_active.append(average_mes)

    messages_passive = []
    messages_passive.append(max_messages_p_df['relationshipCount'].values)
    messages_passive.append(min_messages_p_df['relationshipCount'].values)
    messages_passive.append(average_pas_mes)


    type_of_users = [attacks_active, attacks_passive, trades_active, trades_passive, messages_active, messages_passive]

    for lst in type_of_users:
        objects = ['min', 'avg', 'max']
        y_pos = np.arange(len(objects))
        lst = sorted([float(i) for i in lst])

        plt.bar(y_pos, lst)
        plt.title('Metrics for Active users')
        plt.xticks(y_pos, objects)
        plt.xlabel('metrics')
        plt.ylabel("count")
        plt.show()


def jac_simmilarity(total_attacks, total_trades, total_messages):
    best_in_attacks = total_attacks.nlargest(100, 'relationshipCount').to_numpy()
    best_in_trades = total_trades.nlargest(100, 'relationshipCount').to_numpy()
    best_in_messages = total_messages.nlargest(100, 'relationshipCount').to_numpy()
    jac_attacks_messages = jaccard_score(best_in_attacks[:, 0], best_in_messages[:, 0], average=None)
    jac_attacks_trades = jaccard_score(best_in_attacks[:, 0], best_in_trades[:, 0], average=None)
    jac_messages_trades = jaccard_score(best_in_messages[:, 0], best_in_trades[:, 0], average=None)

    all_sim = []
    all_sim.append(jac_attacks_messages)
    all_sim.append(jac_attacks_trades)
    all_sim.append(jac_messages_trades)

    objects = ['att-tra', 'att-mess', 'mess-tra']
    y_pos = np.arange(len(objects))
    lst = set([float(i) for i in all_sim])
    lst = sorted(lst)
    plt.bar(y_pos, lst)
    plt.title('Metrics for Attacks')
    plt.xticks(y_pos, objects)
    plt.xlabel('metrics')
    plt.ylabel("count")
    plt.show()

    print(j1, j2, j3)
    return j1, j2, j3


def main():
    graph = Graph('127.0.0.1', password='leoleoleo  ')
    print("Read from database")
    attacks = count_active_relationships_for_user("ATTACKS", graph)
    trades = count_active_relationships_for_user("TRADES", graph)
    messages = count_active_relationships_for_user("MESSAGES", graph)
    attacks_p_df = count_passive_relationships_for_user("ATTACKS", graph)
    trades_p_df = count_passive_relationships_for_user("TRADES", graph)
    messages_p_df = count_passive_relationships_for_user("MESSAGES", graph)

    total_attacks = count_pas_active_relations("ATTACKS", graph)
    total_trades = count_pas_active_relations("TRADES", graph)
    total_messages = count_pas_active_relations("MESSAGES", graph)

    print('Calculate Active/Passive moves per user')
    active_sums, passive_sums = total_interactions_per_user(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df)
    print('Calculate in/out degree per user')

    max_attacks, max_trades, max_messages, max_attacks_p_df, max_trades_p_df, max_messages_p_df = \
        max_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df)

    min_attacks, min_trades, min_messages, min_attacks_p_df, min_trades_p_df, min_messages_p_df =  \
        min_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df)

    average_at, average_tr, average_mes, average_pas_at, average_pas_tr, average_pas_mes = \
        average_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df)
    max_sum, min_sum, average_sum, max_sum_passive, min_sum_passive, average_sum_passive =\
        act_pass_metrics(active_sums, passive_sums)
    max_attacks, min_attacks, average_attacks, max_trades, min_trades, average_trades, max_messages, min_messages, \
    average_messages = in_out_degree(total_attacks, total_trades, total_messages)
    plots(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df, active_sums, passive_sums, total_attacks,
          total_trades, total_messages)
    # plots_for_min_max_avg(max_attacks, max_trades, max_messages, max_attacks_p_df, max_trades_p_df, max_messages_p_df,
    #                       min_attacks, min_trades, min_messages, min_attacks_p_df, min_trades_p_df, min_messages_p_df,
    #                       average_at, average_tr, average_mes, average_pas_at, average_pas_tr, average_pas_mes)
    #jac_simmilarity(total_attacks, total_trades, total_messages)


if __name__ == '__main__':
    main()

