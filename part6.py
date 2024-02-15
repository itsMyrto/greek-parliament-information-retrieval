import pickle
import pandas as pd
import spacy
import sqlite3
from copy import deepcopy
from greek_stemmer import stemmer
import re
from time import time
import sys
from helpers.plot import displayPlots
import helpers.databaseCommons as dbCommons

# Global variables that are used often
conn = sqlite3.connect('speeches.db')
try:
    nlp = spacy.load("el_core_news_sm")
except Exception as e:
    print("Please run this: python -m spacy download el_core_news_sm==3.7.0")
    exit(1)
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
    speech_analysis = nlp(speech)
    tokens = []
    for token in speech_analysis:
        """ For each token, strip it of unwanted characters, get it's tag and stem it"""
        tag = str(token.pos_)
        word_token = re.sub(unwantedCharsRegex, "", str(token.text))
        stemmed_word = ""
        
        
        if tag in {"SPACE", "PUNCT", "X", "DET"}:
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

        tokens.append((stemmed_word, tag))
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
    stemmedWordSentiments: dict = pickle.load(open("cacheAndSaved/stemmedWordSentiments.pickle", "rb"))
    
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

    unique_words_for_each_emotion = [0, 0, 0, 0, 0, 0]  # Unique words that contribute to an emotion with the maximum score
    count_all_words = 0
    unique_words = {}  # Unique words and their frequency
    word_for_emotions = [0, 0, 0, 0, 0, 0]  # Words that contribute to an emotion with the maximum score (it counts all occurrences)
    
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

                    count_all_words += 1

                    # Dealing with emotions (if only it was that easy in real life)
                    index = entry["positions"].index(tag)

                    # Skipping neutral words
                    if entry["anger"][index] == entry["disgust"][index] == entry["fear"][index] == entry["happiness"][index] == entry["sadness"][index] == entry["surprise"][index] == 0:
                        continue

                    """Checking if this token is already used from the politician and then increasing its frequency"""
                    if token not in unique_words:
                        unique_words[token] = 1
                    else:
                        freq = unique_words[token]
                        freq += 1
                        unique_words[token] = freq

                    """ Finding the emotion/emotions that the token contributes most """
                    maximum = 0
                    for emotion in ("anger", "disgust", "fear", "happiness", "sadness", "surprise"):
                        if entry[emotion][index] > maximum:
                            maximum = entry[emotion][index]
                        counts["emotions"][emotion] += entry[emotion][index]

                    first_occurrence = False
                    if unique_words[token] == 1:
                        first_occurrence = True

                    """ Increasing the frequency of the emotion/emotions where the token contributes with maximum """
                    ind = 0
                    for emotion in ("anger", "disgust", "fear", "happiness", "sadness", "surprise"):
                        if entry[emotion][index] == maximum:
                            unique_words_for_each_emotion[ind] += 1
                            if first_occurrence:
                                word_for_emotions[ind] += 1
                        ind += 1


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

    # Normalization
    ind = 0
    for emotion in ("anger", "disgust", "fear", "happiness", "sadness", "surprise"):
        counts["emotions"][emotion] *= (word_for_emotions[ind] / unique_words_for_each_emotion[ind])
        counts["emotions"][emotion] /= (count_all_words / 100)
        ind += 1
    return counts


def preFlightCheck():
    # Create the database
    from os.path import isfile, getsize
    # Files to check: speeches.db, cacheAndSaved/inverse_index_catalogue_for_part3.pickle, cacheAndSaved/twMatrix_sparce.pickle, cacheAndSaved/U_s_V.pickle
    
    # Check if initial db has been created
    if not isfile("speeches.db"):
        conn = sqlite3.connect('speeches.db')
        print("Generating speeches.db for the first time, this may take 2-3 mins")
        dbCommons.makeDb(conn)
        conn.close()
    conn = sqlite3.connect('speeches.db')
    
        

if __name__ == "__main__":
    preFlightCheck()
    
    politicians = ["μητσοτακης κωνσταντινου κυριακος", "τσιπρας παυλου αλεξιος"]#, "κουτσουμπας αποστολου δημητριος"]
    #politicians = ["τσιπρας παυλου αλεξιος", "γεωργιαδης αθανασιου σπυριδων-αδωνις", "βαρουφακης γεωργιου γιανης", "κουτσουμπας αποστολου δημητριος", "βελοπουλος ιωσηφ κυριακος", "μητσοτακης κωνσταντινου κυριακος"]
    countsList = []
    
    
    for politician in politicians:
        speechesDF: pd.DataFrame = pd.read_sql_query(f"SELECT * FROM speeches WHERE member_name = \"{politician}\" LIMIT 200", conn)
        
        tokens = speechesToTokens(speechesDF=speechesDF, politician=politician.title())
        
        counts = createCounts(tokens)
        counts["member_name"] = politician.title()
        countsList.append(counts)
    # print(countsList)
    displayPlots(countsList)