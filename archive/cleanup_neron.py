import pandas as pd
from assets.greek_stopwords import STOP_WORDS
from time import time
import re 

import spacy
from greek_stemmer import stemmer
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

keys = {}

def stem_word(word: str) -> str:
    if word in keys:
        return keys[word]
    else:
        keys[word] = word_stemming(word)
        return keys[word] 

unwanted_pattern = re.compile(r'[0-9@#$%^&*()-_=+[\]{};:\'",.<>/?\\|`~!]')
def cleanSpeech(speech: str) -> str:
    
    speech = re.sub(unwanted_pattern, ' ', speech).lower() 
    speech = speech.split()
    
    # print(speech)
    # Converting speeches into bags of words (lists)
    speech = list(filter(lambda word: word != "" and word not in STOP_WORDS and len(word) != 1, speech))
    # speech = " ".join(speech)
    speech = " ".join(list(map(lambda word: stem_word(word), speech)))
    return speech


def cleanup(df: pd.DataFrame) -> pd.DataFrame:


    # Dropping all the rows where the column `member_name` is null, because these speeches don't provide any useful information
    df = df.dropna(subset=['member_name'])

    for i, row in enumerate(df.index):
        df.at[row, "speech"] = cleanSpeech(df.at[row, "speech"])
        
    return df

def processInChunks():
    # Reading the csv file (1280918 total lines, chunksize = how many at once)
    with pd.read_csv("Greek_Parliament_Proceedings_1989_2020.csv", chunksize=100, usecols=["member_name", "sitting_date", "political_party", "speech"]) as cursor:
        for i, chunk in enumerate(cursor):
            start = time()
            df = cleanup(chunk)
            if i == 0:
                df.to_csv("cleaned.csv", index=False, header=True)
            else:
                print(f"Chunk done in {time() - start} seconds")
                df.to_csv("cleaned.csv", mode="a", index=False, header=False)
                start = time()
            # print(f"Chunk {i}/127 done")

if __name__ == "__main__":
    
    processInChunks()
