import pandas as pd

# Read the data from the tsv file
df = pd.read_csv('greek_sentiment_lexicon.tsv', sep='\t')

# Print the first 5 rows of the dataframe
print(df.head())

# Print the number of rows and columns in the dataframe
print(df.shape)
