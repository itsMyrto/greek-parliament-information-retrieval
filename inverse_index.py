import os
import pandas as pd
import math

import cleanup_myrto

if not os.path.isfile("cleaned_data.csv"):
    print("Creating cleaned dataset...")
    cleanup_myrto.clean_dataset()

df = pd.read_csv("cleaned_data.csv")

def get_number_of_docs():
    return len(df)

def create_inverse_index_catalogue() -> dict:

    inverse_index_catalogue = {}

    for index, row in df.iterrows():

        speech = str(row["speech"]) if pd.notna(row["speech"]) else ""

        speech = speech.split(" ")
        doc_id = row["doc_id"]

        for word in speech:
            found = False
            if word in inverse_index_catalogue:
                word_list = inverse_index_catalogue.get(word)
                for i in range(1, len(word_list)):
                    if word_list[i][0] == doc_id:
                        found = True
                        word_list[i][1] += 1
                if not found:
                    word_list[0] += 1
                    word_list.append([doc_id, 1])
                inverse_index_catalogue[word] = word_list
            else:
                word_list = [1, [doc_id, 1]]
                inverse_index_catalogue[word] = word_list

    # print("Catalogue: ", inverse_index_catalogue)
    return inverse_index_catalogue

def calculate_tf_idf_similarity(cleaned_query: list) -> list:

    inverse_index_catalogue = create_inverse_index_catalogue()
    NUMBER_OF_DOCS = get_number_of_docs()
    accumulators = [0] * NUMBER_OF_DOCS
    ld = [0] * NUMBER_OF_DOCS

    for word in cleaned_query:

        if word in inverse_index_catalogue:
            word_list = inverse_index_catalogue[word]
            nt = word_list[0]
            idft = math.log(1 + (NUMBER_OF_DOCS / nt))
            for i in range(1, len(word_list)):
                tf = 1 + math.log(word_list[i][1])
                accumulators[word_list[i][0]] += idft * tf
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
                nt = word_list[0]
                idft = math.log(1 + (NUMBER_OF_DOCS / nt))
                for j in range(1, len(word_list)):
                    if word_list[j][0] == i:
                        tf = 1 + math.log(word_list[j][1])
                        ld[i] += (tf*idft)**2
        accumulators[i] = accumulators[i] / math.sqrt(ld[i])
    return accumulators

