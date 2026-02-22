import pandas as pd 
import re 
from app.config.db_connection import engine

fake_df = pd.read_sql("SELECT * FROM fake_news", engine)
true_df = pd.read_sql("SELECT * FROM true_news", engine)
##scraping_df = pd.read_sql("SELECT * FROM news", engine)

all_df = pd.concat([fake_df, true_df], ignore_index=True)

print(f"Ukupan broj vesti: {len(all_df)}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)    
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)    
    text = re.sub(r"\s+", " ", text).strip()      
    return text

all_df['text'] = all_df['text'].apply(clean_text)

all_df.drop_duplicates(subset='text', inplace=True)
print(f"Broj vesti nakon uklanjanja duplikata: {len(all_df)}")

all_df.to_csv("data/final_training_data.csv", index=False)
print("CSV za trening modela kreiran: data/final_training_data.csv")