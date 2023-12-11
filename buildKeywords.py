import pandas as pd
import sqlite3
import re
import numpy as np
from assets.stopwords import STOPWORDS

stopwords = set(STOPWORDS)


# Bibliography: https://link.springer.com/chapter/10.1007/978-3-540-71701-0_95
def pagerank(bagOfWords: list, num):
    d = 0.85
    window = 3

    # Create an empty graph
    graph = {}
    for i in range(len(bagOfWords)):
        word = bagOfWords[i]
        if word not in graph:
            graph[word] = {"in": 0, "out": 0, "neighbors": [], "score": 1.}

        for j in range(i + 1, min(i + window, len(bagOfWords))):
            neighbor = bagOfWords[j]
            if neighbor not in graph:
                graph[neighbor] = {"in": 0, "out": 0, "neighbors": [], "score": 1.}

            graph[word]["out"] += 1
            graph[neighbor]["in"] += 1
            graph[neighbor]["neighbors"].append(word)


    # Normalizing the scores
    for word in graph.keys():
        graph[word]["score"] = 1. / len(graph)

    # Iterating
    for _ in range(3):
        for word in graph.keys():
            graph[word]["score"] = (1 - d) + d * sum(graph[neighbor]["score"] / graph[neighbor]["out"] for neighbor in graph[word]["neighbors"])

    return sorted(graph, key=lambda x: graph[x]["score"], reverse=True)[:num]


def keywords(speech: str, num: int):
    # pageranks = pagerank(speech.split(), num)
    # print(pageranks)
    return " ".join(pagerank(speech.split(), num))


unwanted_pattern = re.compile(r'[0-9@#$%^&*()-_=+[\]{};:\'",.<>/?\\|`~!]')
accents_translation_table = str.maketrans(
    "άέήίόύώϊΐϋΰὰὲὴὶὸὺὼᾶῆῖῦῶ",
    "αεηιουωιιυυαεηιουωαηιυω"
)


def simplify(x: str):
    # x = re.sub(unwanted_pattern, '', x.lower()).translate(accents_translation_table)
    return re.sub(unwanted_pattern, '', x.lower()).translate(accents_translation_table)


def blobify(x: str):
    return [_ for _ in simplify(x).split() if len(_) > 2 and _ not in stopwords]


def makeDb(conn: sqlite3.Connection):
    # Read 10000 rows from the csv file
    df = pd.read_csv("Greek_Parliament_Proceedings_1989_2020.csv", nrows=10000)
    # Drop the rows where the speaker is unknown. The speaker is found in the `member_id` column
    df = df.dropna(subset=['member_name'])

    # Save in a sql database the speeches of the greek parliament
    df.to_sql("speeches", con=conn, if_exists='replace', index=True)


def makePreProcessedDB(conn: sqlite3.Connection):
    # Getting the unprocessed speeches
    df = pd.read_sql_query("SELECT * FROM speeches", conn)
    # Dropping unnecessary columns
    df.drop(columns=["parliamentary_period", "sitting_date", "member_region", "parliamentary_session",
                     "parliamentary_sitting", "government", "roles", "member_gender"], inplace=True)

    # Maintaining relation to the original dataframe
    df.rename(columns={"index": "ID_0"}, inplace=True)

    print(df.head())

    # Performing preprocessing and keeping the length of the speech which will be useful later
    df["speech"] = df["speech"].apply(lambda x: " ".join(blobify(x)))
    df["speechLength"] = df["speech"].apply(lambda x: len(x.split()))

    # Some speeches are irrelevant (length = 0) so we drop them
    df.drop(df.loc[df['speechLength'] == 0].index, inplace=True)
    print(df.head())

    df.to_sql("processed_speeches", con=conn, if_exists='replace', index=True)


def makeKeywordsDB(conn: sqlite3.Connection):

    # Creating keywords for each speech
    df = pd.read_sql_query("SELECT * FROM processed_speeches", conn)
    df.rename(columns={"index": "ID_1"}, inplace=True)
    df["keywords"] = df["speech"].apply(lambda x: keywords(x, 3))
    df.drop(columns=["speech"], inplace=True)

    df.to_sql("keywords", con=conn, if_exists='replace', index=True)

    # Creating keywords per member
    df = pd.read_sql_query("SELECT * FROM processed_speeches", conn)
    df.groupby("member_name")["speech"].apply(lambda x: " ".join(x)).apply(lambda x: keywords(x, 10)).to_sql("keywords_per_member", con=conn, if_exists='replace', index=True)

    # Creating keywords per political party
    df = pd.read_sql_query("SELECT * FROM processed_speeches", conn)
    df.groupby("political_party")["speech"].apply(lambda x: " ".join(x)).apply(lambda x: keywords(x, 10)).to_sql("keywords_per_party", con=conn, if_exists='replace', index=True)

if __name__ == "__main__":
    conn = sqlite3.connect('speeches.db')

    # makeDb(conn)
    # makePreProcessedDB(conn)
    makeKeywordsDB(conn)
