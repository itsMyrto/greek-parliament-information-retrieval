import os
import numpy as np
from scipy.sparse.linalg import svds
from inverse_index import create_inverse_index_catalogue, get_number_of_docs
from sklearn.cluster import KMeans
import pandas as pd
import pickle


NUMBER_OF_DOCS = get_number_of_docs()
FILEPATH = "/home/myrto/Downloads/Greek_Parliament_Proceedings_1989_2020.csv"

if not os.path.isfile(FILEPATH):
    print("File ", FILEPATH, " not found. Please modify the FILEPATH parameter inside the script.")
    exit(1)

CLUSTERS = 500
CLUSTER_ID = 100
THRESHOLD = 80

def construct_matrix() -> np.array:

    # We load the inverse index catalogue
    if not os.path.isfile("inverse_index.pkl"):
        create_inverse_index_catalogue()

    with open("inverse_index.pkl", 'rb') as file:
        inverse_index_catalogue = pickle.load(file)

    # A 2D empty row-array  is created in order to store the different terms
    terms = np.empty((1, len(inverse_index_catalogue)), dtype=object)

    # A 2D empty column-array is created in order to store the document id's
    documents = np.empty((NUMBER_OF_DOCS, 1), dtype=object)

    # A 2D array full of zeros is created in order to store 1's and 0's
    # 1 if a term exists in a document, 0 otherwise
    # In this matrix the SVD will be applied
    matrix = np.zeros((NUMBER_OF_DOCS, len(inverse_index_catalogue)), np.float32)

    # This block loops through the inverse index catalogue and for
    # each word it finds which documents ids contain the specific word
    # in order to update the matrix array. It also adds each term in the
    # term array and each document id in the documents array
    term_counter = 0
    for term, term_list in inverse_index_catalogue.items():
        terms[0][term_counter] = term
        for i in range(1, len(term_list)):
            document_id = term_list[i][0]
            documents[document_id] = document_id
            matrix[document_id, term_counter] = True
        term_counter += 1

    return matrix, terms, documents


def LSI() -> np.array:
    matrix, terms, documents = construct_matrix()

    # This function returns the top k singular values of the decomposition
    U, S, Vh = svds(matrix, k=THRESHOLD)

    print("Arrays from LSI")
    print(U.shape, U)
    print(S.shape, S)
    print(Vh.shape, Vh)

    # This is a 2D matrix that contains the document representation in a multidimensional space
    # The representation used is the term to concept
    # The formula for this representation is: document_concept = document * V
    # projecting the original data into the space defined by the top singular vectors
    documents_representation = np.matmul(matrix, np.transpose(Vh))

    print("Document representation in multi-dimensional space")
    print(documents_representation.shape, documents_representation)

    kmeans = KMeans(n_clusters=CLUSTERS, random_state=42, n_init="auto")

    # Applying the kmeans algorithm and then save for every point the cluster id it belongs in a new column in the dataframe
    cluster_id = kmeans.fit_predict(documents_representation)

    df_ = pd.read_csv(FILEPATH)
    df_.dropna(subset=['member_name'], inplace=True)
    df_ = df_.reset_index(drop=True)

    print("Printing all the speeches that belong in cluster ", CLUSTER_ID)
    for i in range(0, len(cluster_id)):
        if cluster_id[i] == CLUSTER_ID:
            print(df_.loc[i, "speech"])



LSI()


