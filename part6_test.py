import re

import pandas as pd
import spacy
from greek_stemmer import stemmer

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

    # politician = "βαρουφακης γεωργιου γιανης"
    # df_ = pd.read_csv(FILEPATH)
    #
    # df_ = df_.loc[df_['member_name'] == politician]

    df_ = pd.read_csv("varoufakis.csv")

    # print(df_.head())

    speech_dictionary = {}

    for index, row in df_.iterrows():

        speech_analysis = nlp(row["speech"])
        # print(speech_analysis)

        for token in speech_analysis:

            tag = str(token.pos_)
            word_token = re.sub(re.compile(r'[0-9@#$%^&*()-_=+[\]{};:\'",.<>/?\\|`~!]'), "", str(token.text))
            stemmed_word = ""

            if tag == "SPACE" or tag == "PUNCΤ" or tag == "X" or tag == "DET":
                continue
            elif tag == "NOUN":
                stemmed_word = stemmer.stem_word(word_token, "NNM").lower()
            elif tag == "VERB":
                stemmed_word = stemmer.stem_word(word_token, "VB").lower()
            elif tag == "ADJ" or "ADV":
                stemmed_word = stemmer.stem_word(word_token, "JJM").lower()
            elif tag == "PROPN":
                stemmed_word = stemmer.stem_word(word_token, "PRP").lower()
            else:
                stemmed_word = stemmer.stem_word(word_token, "NNM").lower()

            if stemmed_word in speech_dictionary:
                usage_list = speech_dictionary[stemmed_word]
                if usage_list.count(tag) == 0:
                    usage_list.append(tag)
                    speech_dictionary[stemmed_word] = usage_list
            else:
                speech_dictionary[stemmed_word] = [tag]

    # print(speech_dictionary)


