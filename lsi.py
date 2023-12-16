import numpy as np
from search_engine import create_inverse_index_catalogue

NUMBER_OF_DOCS = 5

# This is the threshold we set for the strongest concepts.
THRESHOLD = 4

def construct_matrix():

    # We load the inverse index catalogue
    inverse_index_catalogue = create_inverse_index_catalogue()

    # A 2D empty row-array  is created in order to store the different terms
    terms = np.empty((1, len(inverse_index_catalogue)), dtype=object)

    # A 2D empty column-array is created in order to store the document id's
    documents = np.empty((NUMBER_OF_DOCS, 1))

    # A 2D array full of zeros is created in order to store 1's and 0's
    # 1 if a term exists in a document, 0 otherwise
    # In this matrix the SVD will be applied
    matrix = np.zeros((NUMBER_OF_DOCS, len(inverse_index_catalogue)))

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
            matrix[document_id, term_counter] = 1
        term_counter += 1

    # Using the SVD function from numpy
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)

    # Finding the top-k concepts that have strength greater than the threshold
    top_k = 0

    for strength in S:
        if strength >= THRESHOLD:
            top_k += 1
        else:
            break

    # This means that there are no concepts with such a big strength as the threshold
    if top_k == 0:
        print("Sorry there is no concept with strength greater than: ", THRESHOLD)
        exit(0)

    # Keeping only the k concepts
    U_k = U[:, :top_k]
    Vh_k = Vh[: top_k, :]
    S_k = S[:top_k]
    S_k = np.diag(S_k)


    # This is a 2D matrix that contains the document representation in a multidimensional space
    # The representation used is the term to concept
    # The formula for this representation is document_concept = document * V
    document_representation = np.matmul(matrix, np.transpose(Vh_k))

    print(document_representation)

construct_matrix()