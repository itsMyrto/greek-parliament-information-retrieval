import pickle
import pandas as pd
import spacy
import sqlite3
from copy import deepcopy
from greek_stemmer import stemmer
import re
from time import time
import sys
from plot import displayPlots

# Global variables that are used often
conn = sqlite3.connect('speeches.db')
nlp = spacy.load("el_core_news_sm")
unwantedCharsRegex = re.compile(r'[0-9@#$%^&*()-_=+[\]{};:\'",.<>/?\\|`~!]')
objectiveValuesLUT = {"SUBJ": 1, "SUBJ-": 0.5, "SUBJ+": 1.5, "OBJ": -1, "BOTH": 0, "POS": 1, "NEG": -1}

# https://stackoverflow.com/a/58602365/9183984
def print_progress_bar(index, total, label):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()

def speechToTokens(speech: str) -> list:
    """ Converting speech to list of tokens (stemmed word) and their tags (adjective, verb etc.)"""
    speech_analysis = nlp(speech.lower())
    tokens = []
    for token in speech_analysis:
        """ For each token, strip it of unwanted characters, get it's tag and stem it"""
        tag = str(token.pos_)
        word_token = re.sub(unwantedCharsRegex, "", str(token.text))
        
        if word_token:
            tokens.append((word_token, tag))
        
    return tokens

def speechesToTokens(speechesDF: pd.DataFrame, politician: str) -> list:
    """ Stemming each speech individually for perf and returning a list of all tokens"""
    tokens = []
    for index, row in speechesDF.iterrows():
        tokens.extend(speechToTokens(row["speech"]))
        print_progress_bar(index, len(speechesDF), f"Processing speech tokens for {politician} ")  # Fancy progress bar ✨
    print()
    return tokens

def createCounts(tokens: list) -> dict:
    """ Creating counts for each emotion and polarity, as well as subjectivity and objectivity """
    
    # Import the stemmedWordSentiments.pickle file
    stemmedWordSentiments: dict = pickle.load(open("cacheAndSaved/stemmedWordSentiments_unstemmed.pickle", "rb"))
    # print(stemmedWordSentiments)
    
    # Initialize counts in a dict as kv pairs (hmmm reminds me of spark kinda)
    # Using dual counters for subjectivity and objectivity, as well as positivity and negativity to count them separately
    counts = {'subjectivity-objectivity': [0, 0],
              'positivity-negativity': [0, 0],
              'emotions': {
              'anger': 0,
              'disgust': 0,
              'fear': 0,
              'happiness': 0,
              'sadness': 0,
              'surprise': 0}}
    
    for token, tag in tokens:
        """ For each token, check if it's in the stemmedWordSentiments dict, if it is, get it's tag and add the emotion and polarity to the counts """
        try:
            if token not in stemmedWordSentiments:
                continue
            else:
                entry = stemmedWordSentiments[token]
                if tag not in entry["positions"]:
                    continue
                else:
                    # Dealing with emotions (if only it was that easy in real life)
                    index = entry["positions"].index(tag)
                    for emotion in ("anger", "disgust", "fear", "happiness", "sadness", "surprise"):
                        counts["emotions"][emotion] += entry[emotion][index]
                    
                    # Dealing with the dual counts
                    subjectivity = objectiveValuesLUT[entry["subjectivity"][index]] #WARNING, CAN RAISE KEYERROR
                    positivity = objectiveValuesLUT[entry["polarity"][index]] #WARNING, CAN RAISE KEYERROR
                    if subjectivity > 0:
                        counts['subjectivity-objectivity'][0] += subjectivity
                    else:
                        counts['subjectivity-objectivity'][1] += -1*subjectivity
                    if positivity > 0:
                        counts['positivity-negativity'][0] += positivity
                    else:
                        counts['positivity-negativity'][1] += -1*positivity
        except Exception as e:
            # print(e)
            continue
                
    return counts


if __name__ == "__main__":
    politicians = ["βελοπουλος ιωσηφ κυριακος"]#, "κουτσουμπας αποστολου δημητριος"]
    # politician = "γεωργιαδης αθανασιου σπυριδων-αδωνις"
    countsList = []
    
    for politician in politicians:
        speechesDF :pd.DataFrame = pd.read_sql_query(f"SELECT * FROM speeches WHERE member_name = \"{politician}\" LIMIT 100", conn)
        
        tokens = speechesToTokens(speechesDF=speechesDF, politician=politician.title())
        tokens.append(("ωμή", "ADJ"))
        print(tokens[-10:])
        
        counts = createCounts(tokens)
        counts["member_name"] = politician.title()
        countsList.append(counts)
    print(countsList)
    displayPlots(countsList)