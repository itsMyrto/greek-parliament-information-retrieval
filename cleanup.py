import pandas as pd
import numpy as np
import string
from assets.stopwords import STOPWORDS

# Reading the csv file (1000 rows for testing purposes)
df = pd.read_csv("Greek_Parliament_Proceedings_1989_2020.csv", nrows=1000)


# Dropping all the rows where the column `member_name` is null, because these speeches don't provide any useful information
df = df.dropna(subset=['member_name'])


# Cleaning up all non alphabetic characters from the column `speech` and converting all characters to lowercase
# Very cursed code, but it works
df["speech"] = df["speech"].transform(lambda x: x.translate(str.maketrans(dict.fromkeys(string.punctuation, ' '))).lower())


# Converting speeches into bags of words (lists)
df["speech"] = df["speech"].transform(lambda x: x.split(" ")).transform(lambda x: np.array(list(filter(lambda y: y != "" and " " not in y, x))))


# Removing stopwords from speeches
df["speech"] = df["speech"].transform(lambda x: np.array(list(filter(lambda y: y not in STOPWORDS, x))))


print(df.head())
