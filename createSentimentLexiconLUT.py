import pandas as pd
import spacy
import sqlite3
from copy import deepcopy
from greek_stemmer import stemmer

conn = sqlite3.connect('speeches.db')
nlp = spacy.load("el_core_news_sm")


tagLUT = {
    "NOUN": "NNM",
    "VERB": "VB",
    "ADJ": "JJM",
    "ADV": "RB",
    "PRON": "PRP",
    "Πρόθημα": "PRP",
    "INTJ": "NNM"
}


def prepWordSentiments(wordSentiments: pd.DataFrame) -> dict:

    def isNaN(subject):
        """Python is weird, nan != nan (True)"""
        return subject != subject

    def cleanWord(word: str) -> str:
        """Lowercasing words and splitting multi-gender to multiple words eg. ωραίος -α -ο -> ωραίος, ωραία, ωραίο"""
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


    def reprocessLookups(lut: dict) -> dict:
        """Making pseudo-objects in a dict for every word, containing it's attributes, from which we're removing empty columns and useless features"""
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
            emotions = {"anger", "disgust", "fear", "happiness", "sadness", "surprise"}
            
            badIndices = set()
            
            # Removing indices which contain NaN values
            for i in range(4):
                for key in newItem:
                    if isNaN(newItem[key][i]):
                        if key == "positions":
                            badIndices.add(i)
                        newItem[key][i] = 0       
                
            
            # combining those with the same positions
            foundPos = set()
            i = 0
            for i, pos in enumerate(newItem["positions"]):
                if pos not in foundPos:
                    foundPos.add(pos)
                    continue
                badIndices.add(i)
            
            # Dropping the bad indices
            for key in newItem:
                newItem[key] = [newItem[key][i] for i in range(4) if i not in badIndices]
                
            lut[item] = newItem 
    
    """Preprocessing pipeline"""
    preprocessLookups(wordSentiments)

    """Working with a dict from the dataframe, so we can do some direct accesses"""
    wordsDict = wordSentiments.set_index('Term').T.to_dict('list')
    secondDict = deepcopy(wordsDict)
    # Hack, to split words with multiple genders
    for word in wordsDict:
        if "," in word:
            words = word.split(",")
            for _word in words:
                secondDict[_word] = wordsDict[word]
            del secondDict[word]
    
    
    reprocessLookups(secondDict)
    return secondDict


def stemWordSentiments(wordSentiments: dict) -> dict:
    """Going through the word sentiments and stemming the words, based on their designation"""
    secondDict = deepcopy(wordSentiments)
    failedList = []
    failedPos = set()
    for key in wordSentiments:
        try:
            newKey = stemmer.stem_word(key, tagLUT[wordSentiments[key]["positions"][0]]).lower()
            del secondDict[key]
            secondDict[newKey] = wordSentiments[key]
        except Exception as e:
            # print(key)
            # print(wordSentiments[key])
            # print(str(e))
            failedList.append((key, wordSentiments[key]["positions"][0]))
            failedPos.add(wordSentiments[key]["positions"][0])
            
            continue

    return secondDict

if __name__ == "__main__":
    # Read the data from the tsv file
    wordSentiments = pd.read_csv('cacheAndSaved/greek_sentiment_lexicon.tsv', sep='\t')

    # Preprocess the word sentiments
    wordSentiments: dict = prepWordSentiments(wordSentiments)
    
    # Stem the word sentiments
    stemmedWordSentiments: dict = stemWordSentiments(wordSentiments)
    
    # Saving the stemmed word sentiments to a file with pickle
    import pickle
    with open("cacheAndSaved/stemmedWordSentiments.pickle", "wb") as f:
        pickle.dump(stemmedWordSentiments, f)
    
        
