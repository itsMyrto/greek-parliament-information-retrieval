import pandas as pd
from inverse_index import get_number_of_docs, calculate_tf_idf_similarity
from cleanup_myrto import word_stemming, remove_unwanted_pattern

TOP_K = 20
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
    temp = list(reversed(sorted(temp)))
    temp = list(dict.fromkeys(temp))[:TOP_K]

    if temp[0] != 0:
        for element in temp:
            if element != 0:
                indexes.append(best_accumulators.index(element))
            else:
                indexes.append(-1)
    print(indexes)
    return indexes

def search_query(query):
    query = query.split(" ")
    cleaned_query = clean_query(query)[1:].split(" ")
    print(cleaned_query)
    similarity_indexes = find_top_k(cleaned_query)
    print("Loading....")
    df_ = pd.read_csv("/home/myrto/Downloads/Greek_Parliament_Proceedings_1989_2020.csv")
    df_.dropna(subset=['member_name'], inplace=True)
    df_ = df_.reset_index(drop=True)

    results = []

    if len(similarity_indexes) != 0:
        for similarity_index in similarity_indexes:
            if similarity_index >= 0:
                title = df_.loc[similarity_index, "sitting_date"] + "-" + df_.loc[similarity_index, "member_name"].upper() + "-" + df_.loc[similarity_index, "political_party"].upper() + ":'" + df_.loc[similarity_index, "speech"][:30] + "..." + "'"
                results.append({"title": title, "content": df_.loc[similarity_index, "speech"]})
    else:
        print("Sorry, nothing found. Please try to rephrase your sentence.")


    return results