import pandas as pd
import math
from cleanup_myrto import word_stemming, remove_unwanted_pattern

TOP_K = 2
NUMBER_OF_DOCS = 5
df = pd.read_csv("cleaned_data.csv")

def create_inverse_index_catalogue():

    inverse_index_catalogue = {}

    for index, row in df.iterrows():

        speech = row["speech"].split(" ")
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


def calculate_tf_idf_similarity(cleaned_query: list) -> list:

    inverse_index_catalogue = create_inverse_index_catalogue()

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
            speech = df["speech"][i].split(" ")
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


search_query()


# TODO: CREATE THE WEB UI FOR THE SEARCH ENGINE







