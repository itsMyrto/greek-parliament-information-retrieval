import pandas as pd
import re
from assets.greek_stopwords import STOP_WORDS
import spacy
from greek_stemmer import stemmer
from time import time
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

"""
This function cleans the speeches which are found in the column `speech` of the dataframe, following these steps:
(1) Drop the rows where the speaker is unknown. The speaker is found in the `member_id` column
(2) For every speech in the greek parliament:
    (2.1) For every word in the speech
        (2.1.1) Remove numbers and symbols
        (2.1.2) Check if the word is empty or a stopword or has length equal to 1. If so then go to step (2.1) for a new word
        (2.1.3) Apply Stemming
        (2.1.4) Add the new word in the new cleaned speech
    (2.2) Replace the speech with the new cleaned speech
    (2.3) Back to (2) to clean an new speech
(3) Return the new data frame    
"""
def clean_data():
    df = pd.read_csv("Greek_Parliament_Proceedings_1989_2020.csv", nrows=20000)
    print("Read the data")
    df = df.dropna(subset=['member_name'])

    start = time()
    count = 0
    unwanted_pattern = re.compile(r'[0-9@#$%^&*()-_=+[\]{};:\'",.<>/?\\|`~!]')
    
    for index, row in df.iterrows():

        speech = row["speech"].split(" ")
        cleaned_speech = ""

        for i in range(0, len(speech)):

            word = speech[i]
            
            cleaned_word = re.sub(unwanted_pattern, '', word)

            if (cleaned_word == "") or (cleaned_word.lower() in STOP_WORDS) or (len(cleaned_word) == 1):
                continue

            cleaned_speech = cleaned_speech + " " + stem_word(cleaned_word).lower()
            # cleaned_speech = cleaned_speech + " " + word_stemming(cleaned_word).lower()

        row["speech"] = cleaned_speech
        # print(row["speech"])
        if count % 100 == 0:
            print(f"chunk done in {time() - start} seconds")
            start = time()
        count += 1
clean_data()