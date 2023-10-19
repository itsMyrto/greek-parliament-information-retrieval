import pandas as pd

df = pd.read_csv("YOUR_CSV_PATH")

# Dropping all the rows where the column `member_name` is null, because these speeches don't provide any useful information
df = df.dropna(subset=['member_name'])

