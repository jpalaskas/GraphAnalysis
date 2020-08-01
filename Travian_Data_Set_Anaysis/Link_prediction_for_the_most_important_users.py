from datetime import datetime

import ujson
from py2neo import Graph


def most_important_users(graph):
    best_in = graph.run('MATCH (n:User)<-[r]-() RETURN n.id, count(r) as count').to_data_frame()
    best_out = graph.run('MATCH (n:User)-[r]->() RETURN n.id, count(r) as count').to_data_frame()

    best_out_degree = [user for user in best_out.nlargest(100, ['count'])['n.id']]
    best_in_degree = [user for user in best_in.nlargest(100, ['count'])['n.id']]
    worst_in_degree = [user for user in best_in.nsmallest(100, ['count'])['n.id']]
    worst_out_degree = [user for user in best_out.nsmallest(100, ['count'])['n.id']]

    user_query = graph.run('''
        MATCH (n:User) RETURN n.id
        ''').to_data_frame()

    nodes = [node for node in user_query['n.id']]

    return best_in, best_out, best_out_degree, best_in_degree, worst_in_degree, worst_out_degree, nodes


def link_prediction(graph, starttime, best_in, best_out, best_out_degree, best_in_degree, worst_in_degree,
                    worst_out_degree, nodes):
    results = {}
    with open('testresultsssss.txt', 'a') as f:
        for s_node in best_out_degree:
            for d_node in nodes:

                if s_node == d_node: continue
                print(d_node)
                a_score_q = graph.run('''
                    MATCH (u1:User) WHERE u1.id={s_id}
                    MATCH (u2:User) WHERE u2.id={d_id}
                    RETURN algo.linkprediction.adamicAdar(u1, u2, 
                    {relationshipQuery: 'ATTACKS'}) AS score
                ''', s_id=s_node, d_id=d_node)
                a_score = a_score_q.data()

                m_score_q = graph.run('''
                    MATCH (u1:User) WHERE u1.id={s_id}
                    MATCH (u2:User) WHERE u2.id={d_id}
                    RETURN algo.linkprediction.adamicAdar(u1, u2, 
                    {relationshipQuery: 'MESSAGES'}) AS score
                ''', s_id=s_node, d_id=d_node)
                m_score = m_score_q.data()

                t_score_q = graph.run('''
                    MATCH (u1:User) WHERE u1.id={s_id}
                    MATCH (u2:User) WHERE u2.id={d_id}
                    RETURN algo.linkprediction.adamicAdar(u1, u2, 
                    {relationshipQuery: 'TRADES'}) AS score
                ''', s_id=s_node, d_id=d_node)
                t_score = t_score_q.data()

                results[(str(s_node), str(d_node))] = {
                    'scores': {
                        'attacks_score': a_score[0]['score'],
                        'messages_score': m_score[0]['score'],
                        'trades_score': t_score[0]['score']
                    }
                }

            f.write(ujson.dumps(results))

    total_time = datetime.now() - starttime
    print(total_time)


def main():
    graph = Graph('127.0.0.1', password='leomamao971')
    starttime = datetime.now()
    best_in, best_out, best_out_degree, best_in_degree, worst_in_degree, worst_out_degree = most_important_users(graph)
    link_prediction(graph, starttime, best_in, best_out, best_out_degree, best_in_degree, worst_in_degree,
                    worst_out_degree)


if __name__ == '__main__':
    main()
