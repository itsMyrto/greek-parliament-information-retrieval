import os
import pandas as pd
import math
import pickle
import dataCleanupPart1


if not os.path.isfile("cleaned_data.csv"):
    print("Creating cleaned dataset...")
    dataCleanupPart1.clean_dataset()

df = pd.read_csv("cleaned_data.csv")

def get_number_of_docs():
    return len(df)

def create_inverse_index_catalogue():
    """
    Create an inverse index catalogue and save it as a pickle file.
    The inverse index catalogue maps words to a list of documents containing the word and their term frequency.
    """
    inverse_index_catalogue = {}

    for index, row in df.iterrows():

        speech = str(row["speech"]) if pd.notna(row["speech"]) else ""

        speech = speech.split(" ")
        doc_id = row["doc_id"]

        for word in speech:
            if word in inverse_index_catalogue:
                word_list = inverse_index_catalogue.get(word)
                if doc_id in word_list:
                    word_list[doc_id] += 1
                else:
                    word_list[doc_id] = 1
                inverse_index_catalogue[word] = word_list
            else:
                inverse_index_catalogue[word] = {doc_id: 1}

        if index % 100000 == 0:
            print(f"Processed {index} documents")
            # print(inverse_index_catalogue)

    print("I am done")

    with open('inverse_index.pkl', 'wb') as file:
        pickle.dump(inverse_index_catalogue, file)

    return

def calculate_tf_idf_similarity(cleaned_query: list) -> list:
    """
    Calculate TF-IDF similarity scores between the query and documents.
    Parameters:
        cleaned_query (list): List of cleaned and stemmed words in the query.
    Returns:
        list: List of TF-IDF similarity scores for each document.
    """

    if not os.path.isfile("inverse_index.pkl"):
        create_inverse_index_catalogue()
        print("Creating the inverse index")

    print("Here")
    with open("inverse_index.pkl", 'rb') as file:
        inverse_index_catalogue = pickle.load(file)

    print("Opened and continuing the work")

    print(len(inverse_index_catalogue))

    NUMBER_OF_DOCS = get_number_of_docs()
    accumulators = [0] * NUMBER_OF_DOCS
    ld = [0] * NUMBER_OF_DOCS
    print("Initialized accumulators")
    
    for word in cleaned_query:

        if word in inverse_index_catalogue:
            word_list = inverse_index_catalogue[word]
            
            nt = len(word_list)
            idft = math.log(1 + (NUMBER_OF_DOCS / nt))
            for doc_id, tf in word_list.items():
                tf = 1 + math.log(tf)
                accumulators[doc_id] += idft * tf
        else:
            continue
    
    for i in range(0, NUMBER_OF_DOCS):
            
        if accumulators[i] == 0:
            continue
        else:
            speech = str(df["speech"][i]) if pd.notna(df["speech"][i]) else ""

            speech = speech.split(" ")

            if len(speech) < 15:
                accumulators[i] = 0
                continue

            for word in speech:
                word_list = inverse_index_catalogue[word]
                
                nt = len(word_list)
                idft = math.log(1 + (NUMBER_OF_DOCS / nt))
                tf = 1 + math.log(word_list[i])
                ld[i] += (tf*idft)**2
        accumulators[i] = accumulators[i] / math.sqrt(ld[i])
        
    return accumulators

if __name__ == "__main__":
    create_inverse_index_catalogue()