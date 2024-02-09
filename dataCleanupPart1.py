import os
import pandas as pd
import re
from assets.greek_stopwords import STOP_WORDS
import spacy
from greek_stemmer import stemmer


FILEPATH = "Greek_Parliament_Proceedings_1989_2020.csv"
try:
    nlp = spacy.load("el_core_news_sm")
except Exception as e:
    print("Please run this: python -m spacy download el_core_news_sm==3.7.0")
    exit(1)


# please run this: python -m spacy download el_core_news_sm==3.7.0
def word_stemming(word: str) -> str:
    """Stem a word based on its part of speech.
    Parameters:
        word (str): Input word.
    Returns:
        str: Stemmed word.
    """
    doc = nlp(word)
    tag = doc[0].pos_
    try:
        if tag == "NOUN":
            return stemmer.stem_word(word, "NNM").lower()
        elif tag == "VERB":
            return stemmer.stem_word(word, "VB").lower()
        elif tag == "ADJ" or "ADV":
            return stemmer.stem_word(word, "JJM").lower()
        elif tag == "PROPN":
            return stemmer.stem_word(word, "PRP").lower()
        else:
            return stemmer.stem_word(word, "NNM").lower()
    except Exception:
        print(word)
        print("Something went wrong, moving on to the next word...")
        return ""


def remove_unwanted_pattern(word: str) -> str:
    """Remove unwanted patterns from a word.
    Parameters:
        word (str): Input word.
    Returns:
        str: Cleaned word.
    """
    unwanted_pattern = re.compile(r'[0-9@#$%^&*()-_=+[\]{};:\'",.<>/?\\|`~!]')
    cleaned_word = re.sub(unwanted_pattern, '', word)
    cleaned_word = re.sub(r'\t', '', cleaned_word)
    if (cleaned_word == " ") or (cleaned_word.lower() in STOP_WORDS) or (len(cleaned_word) == 1) or (
            cleaned_word == ""):
        cleaned_word = ""
    return cleaned_word


def clean_dataset():
    """Clean the dataset and save the cleaned data to a CSV file."""

    dictionary = {}
    cleaned_data = []
    document_id = 0

    if not os.path.isfile(FILEPATH):
        print("File ", FILEPATH, " not found. Please enter a valid path")
        exit(1)

    df = pd.read_csv(FILEPATH)
    print(len(df))
    df = df.dropna(subset=['member_name'])
    df = df.drop(columns=["member_name", "sitting_date", "parliamentary_period", "parliamentary_session",
                          "parliamentary_sitting", "political_party", "government", "member_region", "roles",
                          "member_gender"])

    NUMBER_OF_DOCS = len(df)

    for index, row in df.iterrows():

        speech = row["speech"][1:].replace(u'\xa0', u' ').split(" ")
        cleaned_speech = ""
        # print(speech)

        for i in range(0, len(speech)):

            word = speech[i]
            cleaned_word = remove_unwanted_pattern(word)

            if cleaned_word == "":
                continue

            if cleaned_word in dictionary:
                cleaned_speech = cleaned_speech + " " + dictionary[cleaned_word]
            else:
                stemmed_word = word_stemming(cleaned_word).lower()
                dictionary[cleaned_word] = stemmed_word
                cleaned_speech = cleaned_speech + " " + stemmed_word

        if cleaned_speech == "":
            cleaned_speech = " "

        cleaned_data.append([cleaned_speech, document_id])
        document_id += 1

        if document_id == NUMBER_OF_DOCS:
            new_df = pd.DataFrame(cleaned_data, columns=['speech', 'doc_id'])
            new_df = new_df.astype({"doc_id": "int"})
            new_df = new_df.astype({"speech": "str"})
            new_df.to_csv("cleaned_data.csv")
            break
