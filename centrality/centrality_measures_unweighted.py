from py2neo import Graph
try:
    from centrality_measures import *
except ImportError:
    from .centrality_measures import *

def main():
    graph = Graph('127.0.0.1', password='leomamao971')
    print("Read from database")

    attacks_centrality, attacks_centrality_prob = betweenness_centrality('ATTACKS', graph)
    messages_centrality, messages_centrality_prob = betweenness_centrality('messages', graph)
    trades_centrality, trades_centrality_prob = betweenness_centrality('TRADES', graph)
    graph_betweeness_centrality,graph_betweeness_centrality_prob = betweenness_centrality('', graph)
    out_attacks, in_attacks = degree_centrality("ATTACKS", graph)
    out_trades, in_trades = degree_centrality("TRADES", graph)
    out_messages, in_messages = degree_centrality("messages", graph)

    # pagerank_score = pagerank(" ", graph)
    pagerank_for_attacks_damp, pagerank_for_attacks = pagerank("ATTACKS", graph)
    pagerank_for_trades_damp, pagerank_for_trades = pagerank("TRADES", graph)
    pagerank_for_messages_damp, pagerank_for_messages = pagerank("messages", graph)


    closeness_centrality_attacks = closeness(graph, 'ATTACKS')
    closeness_centrality_messages = closeness(graph, 'TRADES')
    closeness_centrality_trades = closeness(graph, 'messages')
    closeness_centrality_entire_graph = closeness(graph)

    plots_for_measures(attacks_centrality, attacks_centrality_prob, trades_centrality, trades_centrality_prob,
                        messages_centrality, messages_centrality_prob, graph_betweeness_centrality,
                        graph_betweeness_centrality_prob, out_attacks, in_attacks, out_trades, in_trades,
                        out_messages, in_messages, closeness_centrality_attacks, closeness_centrality_trades,
                        closeness_centrality_messages, closeness_centrality_entire_graph,
                        pagerank_for_attacks_damp, pagerank_for_attacks, pagerank_for_trades_damp, pagerank_for_trades,
                        pagerank_for_messages_damp, pagerank_for_messages)
    x=1

if __name__ == '__main__':
    main()