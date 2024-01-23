import pandas as pd
import spacy

FILEPATH = "/home/myrto/Downloads/Greek_Parliament_Proceedings_1989_2020.csv"

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
    df_ = pd.read_csv(FILEPATH)

    df.loc[df['column_name'] == some_value]
