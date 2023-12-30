import pandas as pd
import re
from assets.greek_stopwords import STOP_WORDS
import spacy
from greek_stemmer import stemmer
from time import time
import pickle
nlp = spacy.load("el_core_news_sm")


# please run this: python -m spacy download el_core_news_sm==3.7.1      I didn't know how to add it in the requirements.txt
def word_stemming(word: str) -> str:

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
    # return stemmer.stem_word(word, "NNM")

keys = {}

def stem_word(word: str) -> str:
    if word in keys:
        return keys[word]
    else:
        keys[word] = word_stemming(word)
        return keys[word] 

dictionary = {}

unwanted_pattern = re.compile(r'[0-9@#$%^&*()-_=+[\]{};:\'",.<>/?\\|`~!]')
def remove_unwanted_pattern(word: str) -> str:
    cleaned_word = re.sub(unwanted_pattern, '', word)
    if (cleaned_word == " ") or (cleaned_word.lower() in STOP_WORDS) or (len(cleaned_word) == 1) or (
            cleaned_word == ""):
        cleaned_word = ""
    return cleaned_word
"""
This function cleans the speeches which are found in the column `speech` of the dataframe, following these steps:
(1) Drop the rows where the speaker is unknown. The speaker is found in the `member_id` column
(2) For every speech in the greek parliament:
    (2.1) For every word in the speech
        (2.1.1) Remove numbers and symbols (and/or accents)
        (2.1.2) Check if the word is empty or a stopword or has length equal to 1. If so then go to step (2.1) for a new word
        (2.1.3) Apply Stemming
        (2.1.4) Add the new word in the new cleaned speech
    (2.2) Replace the speech with the new cleaned speech
    (2.3) Back to (2) to clean an new speech
(3) Return the new data frame    
"""
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['member_name'])
    df = df.drop(columns=["member_name", "sitting_date", "parliamentary_period", "parliamentary_session",
                          "parliamentary_sitting", "political_party", "government", "member_region", "roles",
                          "member_gender"])
    
    df.rename(columns={"index": "original_id"}, inplace=True)

    """ Preparing lookup tables for cleaning """
    unwanted_pattern = re.compile(r'[0-9@#$%^&*()-_=+[\]{};:\'",.<>/?\\|`~!]')
    # accents_translation_table = str.maketrans(
    #     "άέήίόύώϊΐϋΰὰὲὴὶὸὺὼᾶῆῖῦῶ",
    #     "αεηιουωιιυυαεηιουωαηιυω"
    # )
    
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
                stemmed_word = stem_word(cleaned_word).lower()
                dictionary[cleaned_word] = stemmed_word
                cleaned_speech = cleaned_speech + " " + stemmed_word

        row["speech"] = cleaned_speech
        
    return df


def processInChunks(filename: str, output: str, nrows: int | None = None, chunksize: int = 100, verbose = bool | None):
    # Reading the csv file (1280918 total lines, chunksize = how many at once)
    with pd.read_csv(filename, chunksize=chunksize, nrows=nrows) as cursor:
        start = time()
        for i, chunk in enumerate(cursor):
            chunkStart = time()
            df = clean_data(chunk)
            if i == 0:
                df.to_csv(output, index=True, header=True)
            else:
                df.to_csv(output, mode="a", index=True, header=False)
            if verbose:
                print(f"Chunk {i} done in {time() - chunkStart} seconds")
                chunkStart = time()
        print(f"Done in {time() - start} seconds")
        
if __name__ == "__main__":
    
    processInChunks(filename="Greek_Parliament_Proceedings_1989_2020.csv", output="cleaned.csv", nrows=10000, chunksize=1000)
    
