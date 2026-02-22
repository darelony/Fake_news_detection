import pandas as pd
from app.config.db_connection import engine

fake_df = pd.read_csv("data/Fake.csv")
fake_df = fake_df[["text"]]
fake_df['label'] = 0

fake_df.to_sql("fake_news", engine, if_exists="replace", index=False)
print(f"{len(fake_df)} FAKE vesti ubačeno u tabelu fake_news!")

true_df = pd.read_csv("data/True.csv")
true_df = true_df[['text']]
true_df['label'] = 1

true_df.to_sql("true_news", engine, if_exists="replace", index=False)
print(f"{len(true_df)} TRUE vesti ubačeno u tabelu true_news!")

scraping_df = pd.read_csv("data/fake_news_latest.csv")
scraping_df = scraping_df[['text']]
scraping_df['label'] = 1  

scraping_df.to_sql("news", engine, if_exists="replace", index=False)
print(f"{len(scraping_df)} scraping vesti ubačeno u tabelu news!")

print("Svi CSV fajlovi su uspešno ubačeni u Postgres!")