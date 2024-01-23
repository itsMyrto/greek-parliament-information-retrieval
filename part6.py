import pandas as pd
import spacy
import sqlite3
conn = sqlite3.connect('speeches.db')

nlp = spacy.load("el_core_news_sm")

def cleanWord(word: str) -> str:

    return word.lower()

def preprocessLookups(df: pd.DataFrame):
    df['Term'] = df['Term'].map(cleanWord)


if __name__ == "__main__":
    # Read the data from the tsv file
    wordSentiments = pd.read_csv('greek_sentiment_lexicon.tsv', sep='\t')

    preprocessLookups(wordSentiments)

    # Print the first 5 rows of the dataframe
    print(wordSentiments.head())

    # Print the number of rows and columns in the dataframe
    print(wordSentiments.shape)

    politician = "βαρβιτσιωτης ιωαννη μιλτιαδης"
    speechesDF :pd.DataFrame = pd.read_sql_query(f"SELECT * FROM speeches where member_name = \"{politician}\"", conn)
    print(speechesDF.head())