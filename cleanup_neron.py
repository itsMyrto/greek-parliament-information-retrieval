import pandas as pd
import numpy as np
import string
from assets.stopwords import STOPWORDS

def cleanup(df: pd.DataFrame) -> pd.DataFrame:

    # keep only the columns we need
    df = df[["member_name", "sitting_date", "political_party", "political_party", "speech"]]


    # Dropping all the rows where the column `member_name` is null, because these speeches don't provide any useful information
    df = df.dropna(subset=['member_name'])


    # Cleaning up all non alphabetic characters from the column `speech` and converting all characters to lowercase
    # Very cursed code, but it works
    df["speech"] = df["speech"].transform(lambda x: x.translate(str.maketrans(dict.fromkeys(string.punctuation, ' '))).lower())


    # Converting speeches into bags of words (lists)
    df["speech"] = df["speech"].transform(lambda x: x.split(" ")).transform(lambda x: np.array(list(filter(lambda y: y != "" and " " not in y, x))))


    # Removing stopwords from speeches
    df["speech"] = df["speech"].transform(lambda x: np.array(list(filter(lambda y: y not in STOPWORDS, x))))

    return df

def processInChunks():
    # Reading the csv file (1280918 total lines, chunksize = how many at once)
    with pd.read_csv("Greek_Parliament_Proceedings_1989_2020.csv", chunksize=10000) as cursor:
        for i, chunk in enumerate(cursor):
            df = cleanup(chunk)
            if i == 0:
                df.to_csv("cleaned.csv", index=False, header=True)
            else:
                df.to_csv("cleaned.csv", mode="a", index=False, header=False)
            print(f"Chunk {i}/127 done")

if __name__ == "__main__":
    
    processInChunks()
