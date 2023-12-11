import pandas as pd
import re
from assets.greek_stopwords import STOP_WORDS
import spacy
from greek_stemmer import stemmer

NUMBER_OF_DOCS = 6

# please run this: python -m spacy download el_core_news_sm==3.7.1      I didn't know how to add it in the requirements.txt
def word_stemming(word: str) -> str:
    nlp = spacy.load("el_core_news_sm")
    doc = nlp(word)
    tag = doc[0].pos_
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


def remove_unwanted_pattern(word: str) -> str:
    unwanted_pattern = re.compile(r'[0-9@#$%^&*()-_=+[\]{};:\'",.<>/?\\|`~!]')
    cleaned_word = re.sub(unwanted_pattern, '', word)
    if (cleaned_word == " ") or (cleaned_word.lower() in STOP_WORDS) or (len(cleaned_word) == 1) or (
            cleaned_word == ""):
        cleaned_word = ""
    return cleaned_word


def clean_dataset():
    dictionary = {}
    document_id = 0
    df = pd.read_csv("/home/myrto/Downloads/Greek_Parliament_Proceedings_1989_2020.csv")
    df = df.dropna(subset=['member_name'])
    df = df.drop(columns=["member_name", "sitting_date", "parliamentary_period", "parliamentary_session",
                          "parliamentary_sitting", "political_party", "government", "member_region", "roles",
                          "member_gender"])

    for index, row in df.iterrows():

        speech = row["speech"].split(" ")
        cleaned_speech = ""

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

        df.loc[index, "speech"] = cleaned_speech[1:]
        df.loc[index, "doc_id"] = document_id
        document_id += 1
        if document_id == NUMBER_OF_DOCS:
            df = df.head(5)
            df = df.astype({"doc_id": "int"})
            df.to_csv("cleaned_data.csv")
            break
