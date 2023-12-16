import pandas as pd
import math
from inverse_index import get_number_of_docs, calculate_tf_idf_similarity
from cleanup_myrto import word_stemming, remove_unwanted_pattern

TOP_K = 2
NUMBER_OF_DOCS = get_number_of_docs()

def clean_query(query: list) -> str:
    cleaned_query = ""

    for word in query:
        cleaned_word = remove_unwanted_pattern(word)
        if cleaned_word == "":
            continue
        else:
            stemmed_word = word_stemming(cleaned_word).lower()
            cleaned_query = cleaned_query + " " + stemmed_word

    return cleaned_query


def find_top_k(cleaned_query: list) -> list:
    best_accumulators = calculate_tf_idf_similarity(cleaned_query)
    indexes = []
    temp = best_accumulators.copy()
    temp = list(reversed(sorted(temp)))[:TOP_K]

    if temp[0] != 0:

        for element in temp:
            if element != 0:
                indexes.append(best_accumulators.index(element))
            else:
                indexes.append(-1)

    return indexes

def search_query():
    query = input("Enter a query: ").split(" ")
    cleaned_query = clean_query(query)[1:].split(" ")
    similarity_indexes = find_top_k(cleaned_query)
    print("Loading....")
    df_ = pd.read_csv("/home/myrto/Downloads/Greek_Parliament_Proceedings_1989_2020.csv")

    if len(similarity_indexes) != 0:
        for similarity_index in similarity_indexes:
            if similarity_index >= 0:
                print(df_.loc[similarity_index, "speech"])
    else:
        print("Sorry, nothing found. Please try to rephrase your sentence.")


# TODO: CREATE THE WEB UI FOR THE SEARCH ENGINE







