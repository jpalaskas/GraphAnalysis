from sklearn.metrics import jaccard_score
from py2neo import Graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from degrees import *
except ImportError:
    from .degrees import *

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

