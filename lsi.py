import numpy as np
from search_engine import create_inverse_index_catalogue

NUMBER_OF_DOCS = 5
THRESHOLD = 4
def construct_matrix():
    inverse_index_catalogue = create_inverse_index_catalogue()
    terms = np.empty((1, len(inverse_index_catalogue)), dtype=object)
    documents = np.empty((NUMBER_OF_DOCS, 1))
    matrix = np.zeros((NUMBER_OF_DOCS, len(inverse_index_catalogue)))


    term_counter = 0
    for term, term_list in inverse_index_catalogue.items():
        terms[0][term_counter] = term
        for i in range(1, len(term_list)):
            document_id = term_list[i][0]
            documents[document_id] = document_id
            matrix[document_id, term_counter] = 1
        term_counter += 1

    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)

    top_k = 0

    for strength in S:
        if strength >= THRESHOLD:
            top_k += 1
        else:
            break

    if top_k == 0:
        print("Sorry there is no concept with strength greater than: ", THRESHOLD)
        exit(0)

    U_k = U[:, :top_k]
    Vh_k = Vh[: top_k, :]
    S_k = S[:top_k]
    S_k = np.diag(S_k)
    S_k_inverse = np.linalg.inv(S_k)

    document_representation = np.matmul(np.matmul(U_k, S_k_inverse), Vh_k)

    print(document_representation)

construct_matrix()