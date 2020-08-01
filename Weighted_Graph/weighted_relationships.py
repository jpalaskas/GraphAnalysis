from py2neo import Graph, Node, Relationship
import os
from datetime import datetime
import pandas as pd
import ujson

dir_path = os.getcwd()

trades = {}
combs = list()

file_cnt = 0
f = open('messages_relations.txt', 'a')
count = 0

for file in os.listdir(os.fsencode(dir_path)):
    filename = os.fsdecode(file)
    if not filename.startswith('message') or not filename.endswith('.csv'): continue
    print('current file --- ', filename)

    df = pd.read_csv(filename, header=None)
    df.columns = ['Timestamp', 'id1', 'id2']

for index, row in df.iterrows():
    # trades[row['id1']] = {
    #     to: row['id2'],
    #     weight: 0,
    #     first_date: None,
    #     second_date: None
    # # }
    if (row['id1'], row['id2']) not in combs:
        ts = int(row['Timestamp'])
        formatted_ts = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        splitted_ts = formatted_ts.split()
        date = splitted_ts[0]
        time = splitted_ts[1]

    combs.append((row['id1'], row['id2']))
    # if (8598, 101) in combs: print('TRUE')

    trades[(row['id1'], row['id2'])] = {
        'weight': combs.count((row['id1'], row['id2'])),
        'first_date': {
            'date': date,
            'time': time
        }
    }

mydict = {'messages': trades}
f.write(ujson.dumps(mydict))
# for tup in combs:
#     trades[tup] = trades.get(tup, 0) + 1

# graph = Graph('127.0.0.1', password='leomamao97')
# tx = graph.begin()
# rel_type = 'MESSAGES'
# count = 9
# for item in trades:
#     graph.run('''MATCH (u1:User),(u2:User)
#                     WHERE u1.id={id1} AND u2.id={id2}
#                     CREATE (u1)-[r:''' + rel_type + \
#                     ''' { weight: {weight}, first_date: {date}, first_time : {time} }]->(u2)''',
#                   id1=int(row['id1']), id2=int(row['id2']),
#                   rel_type=rel_type, weight= trades[item]['weight'],
#                   date=trades[item]['first_date']['date'],
#                   time=trades[item]['first_date']['time'])

#     count +=1
#     if count%1000 == 0 : print(count)
#     # print(item)
#     # print(trades[item])
#     # print(item[0]) ->node
#     # print(item[1]) ->node
#     # print(trades[item]['weight']) ->relationship
#     # print(trades[item]['first_date']['date']) ->relationship
#     # print(trades[item]['first_date']['time']) ->relationship
#     # break

# tx.commit()

