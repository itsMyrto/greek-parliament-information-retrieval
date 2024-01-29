import pandas as pd
from assets.stopwords import STOPWORDS
from time import time
import re
from assets.greek_stopwords import STOP_WORDS
import spacy
from greek_stemmer import stemmer
from time import time
from multiprocessing import Pool

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

def cleanup_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.dropna(subset=['member_name'])

    unwanted_pattern = re.compile(r'[0-9@#$%^&*()-_=+[\]{};:\'",.<>/?\\|`~!]')
    start = time()
    for index, row in chunk.iterrows():
        speech = row["speech"].split(" ")
        cleaned_speech = ""

        for i in range(0, len(speech)):
            word = speech[i]
            cleaned_word = re.sub(unwanted_pattern, '', word)

            if (cleaned_word == "") or (cleaned_word.lower() in STOP_WORDS) or (len(cleaned_word) == 1):
                continue

            cleaned_speech = cleaned_speech + " " + stem_word(cleaned_word).lower()
            # cleaned_speech = cleaned_speech + " " + stemmer.stem_word(word, "NNM").lower()

        row["speech"] = cleaned_speech
    print("Chunk done in ", time() - start, " seconds", flush=True)
    return chunk

def processInChunks():
    # Reading the csv file (1280918 total lines, chunksize = how many at once)
    with pd.read_csv("Greek_Parliament_Proceedings_1989_2020.csv", chunksize=1000, nrows=10000) as cursor:
        pool = Pool()  # Create a multiprocessing pool
        results = pool.map(cleanup_chunk, cursor)  # Apply cleanup_chunk function to each chunk in parallel
        pool.close()
        pool.join()

        df = pd.concat(results)  # Concatenate the cleaned chunks into a single DataFrame

        df.to_csv("cleaned.csv", index=False, header=False)

if __name__ == "__main__":
    processInChunks()
    # save the keys with pickle
    import pickle
    with open('keys.pickle', 'wb') as handle:
        pickle.dump(keys, handle, protocol=pickle.HIGHEST_PROTOCOL)