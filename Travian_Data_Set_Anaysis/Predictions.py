from py2neo import Graph
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

graph = Graph('127.0.0.1', password='leomamao971')


def down_sample(df):
    copy = df.copy()
    zero = Counter(copy.label.values)[0]
    un = Counter(copy.label.values)[1]
    n = zero - un
    copy = copy.drop(copy[copy.label == 0].sample(n=n, random_state=1).index)
    return copy.sample(frac=1)


train_existing_links = graph.run("""
MATCH (u:User)-[:TRADES]->(other:User)
RETURN id(u) AS node1, id(other) AS node2, 1 AS label
""").to_data_frame()

train_missing_links = graph.run("""
MATCH (u:User)
WHERE (user)-[:TRADES]-()
MATCH (user)-[:TRADES*2..3]-(other)
WHERE not((user)-[:User]-(other))
RETURN id(user) AS node1, id(other) AS node2, 0 AS label
""").to_data_frame()
train_missing_links = train_missing_links.drop_duplicates()
training_df = train_missing_links.append(train_existing_links, ignore_index=True)
training_df['label'] = training_df['label'].astype('category')
training_df = down_sample(training_df)

test_existing_links = graph.run("""
MATCH (user:User)-[:TRADES]->(user:User)
RETURN id(user) AS node1, id(user) AS node2, 1 AS label
""").to_data_frame()

test_missing_links = graph.run("""
MATCH (user:User)
WHERE (user)-[:TRADES]-()
MATCH (user)-[:TRADES*2..3]-(other)
WHERE not((user)-[:TRADES]-(other))
RETURN id(user) AS node1, id(other) AS node2, 0 AS label
""").to_data_frame()
test_missing_links = test_missing_links.drop_duplicates()

test_df = test_missing_links.append(test_existing_links, ignore_index=True)
test_df['label'] = test_df['label'].astype('category')
test_df = down_sample(test_df)

classifier = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=0)

def apply_graphy_features(data, rel_type):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
           pair.node2 AS node2,
           algo.linkprediction.commonNeighbors(
               p1, p2, {relationshipQuery: $relType}) AS cn,
           algo.linkprediction.preferentialAttachment(
               p1, p2, {relationshipQuery: $relType}) AS pa,
           algo.linkprediction.totalNeighbors(
               p1, p2, {relationshipQuery: $relType}) AS tn
    """
    pairs = [{"node1": node1, "node2": node2}  for node1,node2 in data[["node1", "node2"]].values.tolist()]
    features = graph.run(query, {"pairs": pairs, "relType": rel_type}).to_data_frame()
    return pd.merge(data, features, on = ["node1", "node2"])

training_df = apply_graphy_features(training_df, "TRADES")

columns = ["cn"]

X = training_df[columns]
y = training_df["label"]
classifier.fit(X, y)

predictions = classifier.predict(test_df[columns])
y_test = test_df["label"]

print("Accuracy", accuracy_score(y_test, predictions))
print("Precision", precision_score(y_test, predictions))
print("Recall", recall_score(y_test, predictions))

sorted(list(zip(columns, classifier.feature_importances_)), key=lambda x: x[1]*-1)