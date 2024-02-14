import os
import numpy as np
from scipy.sparse.linalg import svds
from inverse_index import create_inverse_index_catalogue, get_number_of_docs
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cluster import KMeans
import pandas as pd
import pickle
import random

NUMBER_OF_DOCS = get_number_of_docs()
FILEPATH = "Greek_Parliament_Proceedings_1989_2020.csv"
CLUSTERS = 100
THRESHOLD = 80

if not os.path.isfile(FILEPATH):
    print("File ", FILEPATH, " not found. Please modify the FILEPATH parameter inside the script.")
    exit(1)

def LSI():

    """ We load the inverse index catalogue """
    if not os.path.isfile("inverse_index.pkl"):
        print("Creating the inverse index catalogue...")
        create_inverse_index_catalogue()

    with open("inverse_index.pkl", 'rb') as file:
        inverse_index_catalogue = pickle.load(file)

    """
    A sparse matrix is created to store 1's and 0's
    1 if a term exists in a document, 0 otherwise
    In this matrix, the SVD will be applied
    """
    data = []
    rows = []
    cols = []

    for term_counter, (term, term_dictionary) in enumerate(inverse_index_catalogue.items()):
        for document_id, _ in term_dictionary.items():
            rows.append(document_id)
            cols.append(term_counter)
            data.append(1)

    # Construct the sparse COO matrix
    matrix = coo_matrix((data, (rows, cols)), shape=(NUMBER_OF_DOCS, len(inverse_index_catalogue)), dtype=np.float32)

    del inverse_index_catalogue

    U, S, Vh = svds(matrix, k=THRESHOLD)

    del U, S

    Vh_sparse = csr_matrix(Vh)
    matrix_csr = matrix.tocsr()

    projected_documents = matrix_csr.dot(Vh_sparse.T)

    del Vh, Vh_sparse, matrix, matrix_csr

    np.savez("projected_documents.npz", data=projected_documents.data, indices=projected_documents.indices, indptr=projected_documents.indptr, shape=projected_documents.shape)

    return

def clustering_speeches():

    if not os.path.isfile("projected_documents.npz"):
        LSI()

    loaded_data = np.load("projected_documents.npz")
    projected_documents = csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])
    random_subset_size = projected_documents.shape[0] // 2

    # Randomly sample a subset for K-means clustering
    random_indices = random.sample(range(projected_documents.shape[0]), random_subset_size)
    random_subset = projected_documents[random_indices]

    # Run K-means clustering on the random subset
    kmeans = KMeans(n_clusters=CLUSTERS, random_state=0)
    random_labels = kmeans.fit_predict(random_subset)
    cluster_centers = kmeans.cluster_centers_

    # Create an array with vectors corresponding to the remaining indices
    # Get the indices of documents not in the random subset
    remaining_indices = np.setdiff1d(np.arange(projected_documents.shape[0]), random_indices)

    # Save results to a file
    np.savez("kmeans_results.npz", random_labels=random_labels, cluster_centers=cluster_centers, random_indices=random_indices, remaining_indices=remaining_indices)

    # Load the kmeans results from the file
    results = np.load("kmeans_results.npz")

    # Access the random_labels and cluster_centers arrays
    random_labels = results['random_labels']
    cluster_centers = results['cluster_centers']
    random_indices = results['random_indices']
    remaining_indices = results['remaining_indices']

    # Initialize dictionary to store document IDs and cluster assignments
    cluster_document_info = {i: [] for i in range(CLUSTERS)}
    for i in range(random_subset_size):
        cluster_document_info[random_labels[i]].append(random_indices[i])


    # Create an array with vectors corresponding to the remaining indices
    remaining_vectors = projected_documents[remaining_indices]

    # Compute Manhattan distance for each remaining document to each cluster center
    i = 0
    for doc_id, vector in zip(remaining_indices, remaining_vectors):
        min_distance = float('inf')
        min_cluster = -1
        for j, center in enumerate(cluster_centers):
            distance = np.sum(np.abs(vector - center))  # Manhattan distance
            if distance < min_distance:
                min_distance = distance
                min_cluster = j
        cluster_document_info[min_cluster].append(doc_id)
        i += 1
        if i % 100000 == 0:
            print("Processed ", i, " documents")

    # Print the dictionary containing document IDs and cluster assignments
    for cluster, info in cluster_document_info.items():
        print("Cluster", cluster, ":")
        print("  Document IDs:", info)

    with open('final_clustering_results.pkl', 'wb') as file:
        pickle.dump(cluster_document_info, file)

    return


def print_clusters():

    cluster_id = 13

    if not os.path.isfile("final_clustering_results.pkl"):
        clustering_speeches()

    with open("final_clustering_results.pkl", 'rb') as file:
        cluster_document_info = pickle.load(file)

    random_cluster = cluster_document_info[cluster_id]

    df_ = pd.read_csv(FILEPATH)
    df_.dropna(subset=['member_name'], inplace=True)
    df_ = df_.reset_index(drop=True)

    for index in random_cluster:
        print(df_.loc[index, "speech"])

    return


print_clusters()


