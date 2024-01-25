import pandas as pd
import sqlite3
import re
import numpy as np
from assets.stopwords import STOPWORDS
import sys
import argparse



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


def extractKeywords(speech: str, num: int):
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


def makeDb(conn: sqlite3.Connection, max_rows: int = None):
    # Read 10000 rows from the csv file
    if max_rows:
        df = pd.read_csv("Greek_Parliament_Proceedings_1989_2020.csv", nrows=max_rows)
    else:
        df = pd.read_csv("Greek_Parliament_Proceedings_1989_2020.csv")
    # Drop the rows where the speaker is unknown. The speaker is found in the `member_id` column
    df = df.dropna(subset=['member_name'])
    df["sitting_date"] = pd.to_datetime(df["sitting_date"], format="%d/%m/%Y")

    # Save in a sql database the speeches of the greek parliament
    df.to_sql("speeches", con=conn, if_exists='replace', index=True)


def makePreProcessedDB(conn: sqlite3.Connection):
    # Getting the unprocessed speeches
    df = pd.read_sql_query("SELECT * FROM speeches", conn)
    # Dropping unnecessary columns
    df.drop(columns=["parliamentary_period", "member_region", "parliamentary_session",
                     "parliamentary_sitting", "government", "roles", "member_gender"], inplace=True)

    # Maintaining relation to the original dataframe
    df.rename(columns={"index": "ID_0"}, inplace=True)


    # Performing preprocessing and keeping the length of the speech which will be useful later
    df["speech"] = df["speech"].apply(lambda x: " ".join(blobify(x)))
    df["speechLength"] = df["speech"].apply(lambda x: len(x.split()))

    # Some speeches are irrelevant (length = 0) so we drop them
    df.drop(df.loc[df['speechLength'] == 0].index, inplace=True)

    df.to_sql("processed_speeches", con=conn, if_exists='replace', index=True)


def makeKeywordsDB(conn: sqlite3.Connection):

    # Creating keywords for each speech
    df = pd.read_sql_query("SELECT * FROM processed_speeches", conn)
    df.rename(columns={"index": "ID_1"}, inplace=True)
    df["keywords"] = df["speech"].apply(lambda x: extractKeywords(x, 3))
    df.drop(columns=["speech"], inplace=True)

    df.to_sql("keywords", con=conn, if_exists='replace', index=True)

    # Creating keywords per member
    df = pd.read_sql_query("SELECT * FROM processed_speeches", conn)
    df.groupby("member_name")["speech"].apply(lambda x: " ".join(x)).apply(lambda x: extractKeywords(x, 10)).to_sql("keywords_per_member", con=conn, if_exists='replace', index=True)

    # Creating keywords per political party
    df = pd.read_sql_query("SELECT * FROM processed_speeches", conn)
    df.groupby("political_party")["speech"].apply(lambda x: " ".join(x)).apply(lambda x: extractKeywords(x, 10)).to_sql("keywords_per_party", con=conn, if_exists='replace', index=True)


def makeKeywordsFromDF(df: pd.DataFrame):
    keywords = extractKeywords(" ".join(df["speech"].tolist()), 10)
    return keywords

# Handling the date dependent keyword queries
def getKeywordQueryByDate(conn: sqlite3.Connection, mode: str = "speeches", entity: str | None = None, output: str = "console"):
    
    # Splitting based on the mode
    
    if mode == "speeches":
        # Getting the speeches
        df = pd.read_sql_query("SELECT * FROM processed_speeches", conn)
        # Changing dates to datetime objects so we can group by year
        df["sitting_date"] = pd.to_datetime(df["sitting_date"])
        df = df.groupby(df["sitting_date"].dt.year)
        # Creating a results dataframe
        newDF = pd.DataFrame(columns=["year", "keywords"])
        for year, group in df:
            keywords = makeKeywordsFromDF(group)
            newEntry = pd.DataFrame({"year": [year], "keywords": [keywords]})
            newDF = pd.concat([newDF, newEntry], ignore_index=True)
        if output == "console":
            print(newDF)
        else:
            newDF.to_csv(output, index=False)
    
    if mode == "members":
        # Getting the member speeches where the member is the entity
        df = pd.read_sql_query(f"SELECT * FROM processed_speeches WHERE member_name = {entity.lower()}", conn)
        # Changing dates to datetime objects so we can group by year
        df["sitting_date"] = pd.to_datetime(df["sitting_date"])
        df = df.groupby(df["sitting_date"].dt.year)
        # Creating a results dataframe
        newDF = pd.DataFrame(columns=["year", "keywords"])
        for year, group in df:
            keywords = makeKeywordsFromDF(group)
            newEntry = pd.DataFrame({"year": [year], "keywords": [keywords]})
            newDF = pd.concat([newDF, newEntry], ignore_index=True)
        if output == "console":
            print(newDF)
        else:
            newDF.to_csv(output, index=False)
    
    if mode == "parties":
        # Getting the party speeches where the party is the entity
        df = pd.read_sql_query(f"SELECT * FROM processed_speeches WHERE political_party = {entity.lower()}", conn)
        # Changing dates to datetime objects so we can group by year
        df["sitting_date"] = pd.to_datetime(df["sitting_date"])
        df = df.groupby(df["sitting_date"].dt.year)
        # Creating a results dataframe
        newDF = pd.DataFrame(columns=["year", "keywords"])
        for year, group in df:
            keywords = makeKeywordsFromDF(group)
            newEntry = pd.DataFrame({"year": [year], "keywords": [keywords]})
            newDF = pd.concat([newDF, newEntry], ignore_index=True)
        if output == "console":
            print(newDF)
        else:
            newDF.to_csv(output, index=False)



if __name__ == "__main__" and False:
    with sqlite3.connect('speeches.db') as conn:
    #    makeKeywordsDB(conn)
        getKeywordQueryByDate(conn, mode="speeches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line utility for database initialization and keyword processing.")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum number of rows for database initialization (defaults to all)")
    parser.add_argument("--initialize-database", action="store_true", help="Initialize the database with the specified max rows")
    parser.add_argument("--process-keywords", action="store_true", help="Process and build keywords")
    parser.add_argument("--gather-politician-keywords", type=str, help="Gather keywords for a politician by providing the politician's name")
    parser.add_argument("--gather-party-keywords", type=str, help="Gather keywords for a party by providing the party's name")
    parser.add_argument("--gather-total-keywords", action="store_true", help="Gather total keywords")
    
    parser.add_argument("--output", type=str, default="console", help="Output results to a file (defaults to console)")
    
    args = parser.parse_args()
    
    if args.initialize_database:
        if args.max_rows:
            max_rows = args.max_rows
        else:
            max_rows = None
        with sqlite3.connect('speeches.db') as conn:
            makeDb(conn)
            makePreProcessedDB(conn)
    if args.process_keywords:
        with sqlite3.connect('speeches.db') as conn:
            conn = sqlite3.connect('speeches.db')
            makeKeywordsDB(conn)
    if args.gather_politician_keywords:
        with sqlite3.connect('speeches.db') as conn:
            getKeywordQueryByDate(conn, mode="members", entity=args.gather_politician_keywords, output=args.output)
    if args.gather_party_keywords:
        with sqlite3.connect('speeches.db') as conn:
            getKeywordQueryByDate(conn, mode="parties", entity=args.gather_party_keywords, output=args.output)
    if args.gather_total_keywords:
        with sqlite3.connect('speeches.db') as conn:
            getKeywordQueryByDate(conn, mode="speeches", output=args.output)
        