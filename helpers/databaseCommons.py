import pandas as pd
import sqlite3
import re
import numpy as np
from assets.stopwords import STOPWORDS
import pickle
from time import time
FILENAME = "Greek_Parliament_Proceedings_1989_2020.csv"
stopwords = set(STOPWORDS)
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
        df = pd.read_csv(FILENAME, nrows=max_rows)
    else:
        df = pd.read_csv(FILENAME)
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