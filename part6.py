import pandas as pd
import spacy
import sqlite3
from copy import deepcopy
conn = sqlite3.connect('speeches.db')
nlp = spacy.load("el_core_news_sm")




def cleanWord(word: str) -> str:
    word = word.lower()
    words = []
    if " " in word:
        components = word.split(" ")
        word = components[0]
        extraComponents = components[1].replace(" ", "").split("-")
        words = [word] + [word[:-2] + ending for ending in extraComponents[1:]]
        word = ",".join(words)
    return word

def preprocessLookups(df: pd.DataFrame):
    df['Term'] = df['Term'].map(cleanWord)


def processSpeech(speech: str) -> str:
    wordList = speech.split(" ")[:5000]
    doc = nlp(speech)
    print([token.lemma_ for token in doc])

def reprocessLookups(lut: dict) -> dict:
    for item in lut:
        """{Term} [POS1 POS2 POS3 POS4	Subjectivity1 Subjectivity2	Subjectivity3 Subjectivity4	Polarity1 Polarity2	Polarity3 Polarity4	Anger1 Anger2 Anger3 Anger4 Disgust1 Disgust2 Disgust3 Disgust4	Fear1 Fear2	Fear3 Fear4	Happiness1 Happiness2 Happiness3 Happiness4 Sadness1 Sadness2 Sadness3 Sadness4	Surprise1 Surprise2	Surprise3 Surprise4	Aditional1 Aditional2 Aditional3 Aditional4 Comments1 Comments2	Comments3 Comments4"""
        """ Last 8 indices are useless """
        newItem = {}
        lut[item] = lut[item][:-8]
        newItem["positions"] = lut[item][:4]
        newItem["subjectivity"] = lut[item][4:8]
        newItem["polarity"] = lut[item][8:12]
        newItem["anger"] = lut[item][12:16]
        newItem["disgust"] = lut[item][16:20]
        newItem["fear"] = lut[item][20:24]
        newItem["happiness"] = lut[item][24:28]
        newItem["sadness"] = lut[item][28:32]
        newItem["surprise"] = lut[item][32:36]
        print(type(newItem["happiness"][1]))
        nan = float("nan")
        badIndices = []
        for i in range(4):
            for key in newItem:
                if newItem[key][i] == nan:
                    badIndices.append(i)
        print(badIndices)
        exit(1)
        # for i in range(4):
        #     if newItem["positions"][i] == nan:
        # lut[item] = newItem
        
if __name__ == "__main__":
    # Read the data from the tsv file
    wordSentiments = pd.read_csv('greek_sentiment_lexicon.tsv', sep='\t')

    preprocessLookups(wordSentiments)
    sentimentalWordsSet = set(wordSentiments['Term'].tolist())
    # print(sentimentalWordsSet)

    wordsDict = wordSentiments.set_index('Term').T.to_dict('list')
    secondDict = deepcopy(wordsDict)
    for word in wordsDict:
        if "," in word:
            words = word.split(",")
            for _word in words:
                secondDict[_word] = wordsDict[word]
            del secondDict[word]
    # print(secondDict)
    reprocessLookups(secondDict)
    # Print the first 5 rows of the dataframe
    # print(wordSentiments.head())

    # Print the number of rows and columns in the dataframe
    # print(wordSentiments.shape)

    # politician = "βελοπουλος ιωσηφ κυριακος"
    # speechesDF :pd.DataFrame = pd.read_sql_query(f"SELECT * FROM speeches where member_name = \"{politician}\"", conn)
    
    # # combining all the speeches into a large string
    # allText = " ".join(speechesDF['speech'].tolist())
    # print(speechesDF.shape)
    # processSpeech(allText)