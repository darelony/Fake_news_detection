import pandas as pd
from app.config.db_connection import engine

df = pd.read_sql("SELECT * FROM news", engine)

df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace(r"http\S+", "", regex=True)
df['text'] = df['text'].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
df = df.drop_duplicates(subset=['text'])

df.to_csv("data/fake_news_latest.csv", index=False)

print("ETL zavrsio, CSV spreman za model")