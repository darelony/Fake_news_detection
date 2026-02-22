import feedparser
import pandas as pd 
from app.config.db_connection import engine

url = "https://feeds.bbci.co.uk/news/rss.xml"
feed = feedparser.parse(url)

data = []
for entry in feed.entries:
    text = entry.title + " " + entry.summary
    data.append({"text": text})

df = pd.DataFrame(data)

df.to_sql("news", engine, if_exists="append", index=False)
print("Scraper zavrsio i podaci su ubaceni u Postgres!")