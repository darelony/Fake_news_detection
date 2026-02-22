    # Fake News Pipeline

    Ovaj projekat implementira ETL pipeline i pripremu podataka za detekciju lažnih vesti.  
    Cilj je prikupiti nove vesti, spojiti ih sa Kaggle dataset-om i pripremiti CSV za trening ML modela.

    Ovaj projekat implementira automatizovani pipeline za:
        - 📰 Prikupljanje novih vesti (web scraping)
        - 🗄️ ETL proces i skladištenje u PostgreSQL bazu
        - 🔄 Spajanje sa originalnim Kaggle dataset-om
        - 🧠 Re-treniranje ML modela na proširenom dataset-u
        - 🎯 **Cilj:** Rešavanje temporal domain shift problema iz glavnog projekta

    ## Struktura projekta
    fake_news_pipeline/
    │
    ├── app/
    │ ├── scraper/ # skripte za scraping novonabavljenih vesti
    │ ├── etl/ # ETL skripte za čišćenje i pripremu podataka
    │ ├── training/ # skripte za re-trening modela
    │ └── config/ # konfiguracija i konekcija ka Postgres
    │
    ├── database/ # Docker volume za Postgres
    ├── data/ # CSV fajlovi: Fake.csv, True.csv, fake_news_latest.csv
    ├── models/ # sačuvani modeli i TF-IDF vektorizatori
    ├── tests/ # testovi (opciono)
    │
    ├── docker-compose.yml # Postgres container
    ├── Dockerfile
    ├── requirements.txt
    ├── .env
    └── README.md

    ---

    ## Tehnologije i biblioteke

    - Python 3.11  
    - Pandas  
    - SQLAlchemy  
    - psycopg2  
    - scikit-learn  
    - Docker / Docker Compose  

    ---

    ## Instalacija i pokretanje

    1. Kloniraj repozitorijum:

    ```bash
    git clone <repo-link>
    Pokreni virtualno okruženje i instaliraj zavisnosti:

    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    .\venv\Scripts\activate    # Windows
    pip install -r requirements.txt


    Pokreni Docker container za Postgres:

    docker compose up -d


    Pokreni scraper (ako želiš nove vesti):

    python -m app.scraper.scraper


    Ubaci Kaggle dataset u Postgres:

    python -m app.etl.load_csv_to_postgres


    Napravi finalni CSV za trening:

    python -m app.etl.create_training_csv


    Re-treniraj ML model:

    python -m app.training.train_model

    Tabele u bazi
    Tabela	Sadržaj	Label
    fake_news	Kaggle FAKE vesti	1
    true_news	Kaggle TRUE vesti	0
    news	Novonabavljene vesti preko scraping-a	0

    ## Predikcija novih vesti 
    import joblib

    model = joblib.load("models/fake_news_model_updated.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    texts = ["Primer lažne vesti", "Primer prave vesti"]
    pred = model.predict(vectorizer.transform(texts))

    print(pred)  # 0 = TRUE, 1 = FAKE
    Napomene
    - Originalni model u folderu Fake_news_detection ostaje netaknut.

    - Novi model (fake_news_model_updated.pkl) se koristi samo za nove podatke.

    - Scraping novih vesti može biti unapređen i automatizovan periodički.

    Autor
    Darko Matic