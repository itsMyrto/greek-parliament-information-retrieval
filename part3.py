import pandas as pd
import sqlite3
import re
import numpy as np
from assets.stopwords import STOPWORDS
import pickle
from time import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

stopwords = set(STOPWORDS)


def makeDb(conn: sqlite3.Connection):
    # Read 10000 rows from the csv file
    df = pd.read_csv("Greek_Parliament_Proceedings_1989_2020.csv", nrows=10000)
    # Drop the rows where the speaker is unknown. The speaker is found in the `member_id` column
    df = df.dropna(how='any')

    # Save in a sql database the speeches of the greek parliament
    df.to_sql("speeches", con=conn, if_exists='replace', index=True)


def makeGroupedDB(conn: sqlite3.Connection):
    df = pd.read_sql_query("SELECT * FROM processed_speeches", conn)
    df.groupby("member_name")["speech"].apply(lambda x: " ".join(x)).to_sql("ProcessedSpeechesPerMember", con=conn, if_exists='replace', index=True)



def createInverseIndex(conn: sqlite3.Connection):
    start = time()
    df = pd.read_sql_query("SELECT * FROM ProcessedSpeechesPerMember", conn)
    print(f"Reading from db took {time() - start} seconds")
    
    # Building inverse index for terms
    start = time()
    inverse_index_catalogue = {}
    for index, row in df.iterrows():
        for term in row["speech"].split():
            if term not in inverse_index_catalogue:
                inverse_index_catalogue[term] = [(index, 1)]
            elif inverse_index_catalogue[term][-1] != index:
                inverse_index_catalogue[term].append((index, 1))
            else:
                inverse_index_catalogue[term][-1] = (index, inverse_index_catalogue[term][-1][1] + 1)
        if index % 100 == 0:
            print(f"Processed {index} documents")
            
    print(f"Building inverse index took {time() - start} seconds")
    # Save the inverse index catalogue in a file with pickle
    with open("inverse_index_catalogue_for_part3.pickle", "wb") as f:
        pickle.dump(inverse_index_catalogue, f)
    
    

def createTWMatrix(conn: sqlite3.Connection):
    start = time()
    inverse_index_catalogue = pickle.load(open("inverse_index_catalogue_for_part3.pickle", "rb"))
    df = pd.read_sql_query("SELECT * FROM ProcessedSpeechesPerMember", conn)
    print(f"Reading from pickle took {time() - start} seconds")
    
    start = time()
    twMatrix = np.zeros((len(df), len(inverse_index_catalogue)))
    for index, key in enumerate(inverse_index_catalogue.keys()):
        for doc_id, tf in inverse_index_catalogue[key]:
            twMatrix[doc_id][index] = tf
        if index % 333000 == 0:
            print(f"Processed {index} terms")
    print(f"Building term-document matrix took {time() - start} seconds")
    twMatrix_sparce = csr_matrix(twMatrix)
    
    with open("twMatrix_sparce.pickle", "wb") as f:
        pickle.dump(twMatrix_sparce, f)


def createSVDMatrix(conn: sqlite3.Connection):
    start = time()
    twMatrix_sparce = pickle.load(open("twMatrix_sparce.pickle", "rb"))
    print(f"Reading from pickle took {time() - start} seconds")
    
    start = time()
    U, s, V = svds(twMatrix_sparce, k=100)
    print(U.shape, s.shape, V.shape)
    print(f"SVD took {time() - start} seconds")
    
    with open("U_s_V.pickle", "wb") as f:
        pickle.dump((U, s, V), f)


def cosineMatMul(A: np.matrix, B: np.matrix) -> np.matrix:
    return np.matmul(A, B.T) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)).reshape(-1, 1)


def findTopKSimilarMembers(conn: sqlite3.Connection, k: int):
    # Reading the SVD decomposition from pickle
    start = time()
    U, s, V = pickle.load(open("U_s_V.pickle", "rb"))
    print(f"Reading from pickle took {time() - start} seconds")
    
    # Reading the speeches per member dataframe from the db
    start = time()
    df = pd.read_sql_query("SELECT * FROM ProcessedSpeechesPerMember", conn)
    print(f"Reading from db took {time() - start} seconds")
    
    
    # Normalizing each column of U
    start = time()
    norms = np.linalg.norm(U, axis=0)
    U = U / norms
    print(f"Normalizing U took {time() - start} seconds")
    print()
    
    # Inner product to find similarity between pairs of members (and matrix cleanup)
    start = time()
    # simMatrix = cosineMatMul(U, U)
    simMatrix = np.matmul(U, U.T)
    memberSimilarityMatrix = np.triu(simMatrix)
    np.fill_diagonal(memberSimilarityMatrix, 0)
    
    # Finding the top k pairs of members
    topKPairs = np.unravel_index(np.argsort(memberSimilarityMatrix.ravel())[-k:], memberSimilarityMatrix.shape)
    listOfPairs = list(zip(topKPairs[0], topKPairs[1]))
    for pair in listOfPairs:
        print(f'"{df.iloc[pair[0]]["member_name"]}", "{df.iloc[pair[1]]["member_name"]}"')
    
    print(f"\nFinding top {k} similar members took {time() - start} seconds")
    

if __name__ == "__main__":
    conn = sqlite3.connect('speeches.db')
    
    # TODO: Change so it uses Myrto's cleandata instead of mine
    
    # makeDb(conn)
    # makeGroupedDB(conn)
    # makeShinglesDB(conn)
    # createInverseIndex(conn)
    # createTWMatrix(conn)
    # createSVDMatrix(conn)
    findTopKSimilarMembers(conn, 5)
    
    
"""
Journal:
    - Cosine similarity for the member similarity has the issue of matching members with vast differences in the bulk of their speeches, thus not capturing the similarity of the members.
    Thus we use the inner product of the normalized columns of U to find the similarity between the members.
    
    - We're grouping the speeches per member into a single document,
    then we create the inverse index for the terms in the speeches, which will help us create the doc/term matrix.
    We compress it as sparse.
    Then we use SVD to decompose the matrix into U, s, V. This helps us capture the main topics of discussion in the parliament and their
    relation to the members in the U matrix. Scipi handles selecting the most importan topics for us.
    Since the members are relatively few and there's little chance of their count increasing by a whole order of magnitude, we can use more intensive methods to find the similarity between them
    such as a full inner product with O(n^2) complexity.
    This is a technique inspired by RBF networks used in Neural Networks.
"""