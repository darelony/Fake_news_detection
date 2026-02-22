import streamlit as st
import re
import joblib
import json
import os
from datetime import datetime

# Funkcija za čišćenje teksta
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ✅ Učitaj model iz SHARED folder
@st.cache_resource
def load_model():
    # Probaj prvo shared models folder
    SHARED_MODELS = "../shared_models"
    LOCAL_MODELS = "."  # Fallback
    
    try:
        # Pokušaj učitati iz shared foldera
        model = joblib.load(f"{SHARED_MODELS}/lr_model.pkl")
        vectorizer = joblib.load(f"{SHARED_MODELS}/tfidf_vectorizer.pkl")
        
        # Učitaj metadata
        try:
            with open(f"{SHARED_MODELS}/model_metadata.json", "r") as f:
                metadata = json.load(f)
        except:
            metadata = {"version": "1.0", "source": "shared"}
        
        st.success(f"✅ Model loaded from SHARED folder (Version {metadata.get('version', 'N/A')})")
        return model, vectorizer, metadata, "shared"
        
    except Exception as e:
        # Fallback - učitaj lokalne modele
        st.warning(f"⚠️ Shared models not found, using local models")
        try:
            model = joblib.load('lr_model.pkl')
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
            metadata = {"version": "1.0", "source": "local"}
            return model, vectorizer, metadata, "local"
        except Exception as e2:
            st.error(f"❌ Error loading models: {e2}")
            st.stop()

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("🕵️ Fake News Detector")
st.markdown("### Detektuj lažne vesti pomoću AI modela!")
st.markdown("---")

# Učitaj model
model, vectorizer, metadata, source = load_model()

# Input polje
st.subheader("📝 Unesi tekst vesti:")
news_text = st.text_area(
    "Kopiraj vest ovde (mora biti na ENGLESKOM):",
    height=200,
    placeholder="Paste your news article here..."
)

# Primer vesti
with st.expander("💡 Primeri vesti za testiranje"):
    st.markdown("""
    **FAKE vest primer:**
    ```
    BREAKING NEWS: Hollywood celebrity caught living secret double life as 
    alien ambassador! Shocking photos reveal the truth they don't want you 
    to see! Sources claim late-night meetings with extraterrestrial beings. 
    Share this before it gets deleted!
    ```
    
    **TRUE vest primer:**
    ```
    WASHINGTON (Reuters) - The United States government announced on Monday 
    a comprehensive new education reform policy aimed at improving STEM education 
    in public schools across the nation. The Department of Education released 
    a detailed report outlining the key components of the initiative.
    ```
    
    **💡 Tip:** Model radi najbolje sa dugim vestima (100+ reči). Kratke vesti mogu biti netačno klasifikovane.
    """)

# Dugme za analizu
if st.button("🔍 Analiziraj Vest", type="primary"):
    if news_text.strip() == "":
        st.warning("⚠️ Molim te unesi tekst vesti!")
    else:
        # Check minimum length
        word_count = len(news_text.split())
        if word_count < 20:
            st.warning(f"⚠️ Vest je prekratka ({word_count} reči). Za najbolje rezultate, vest treba da ima najmanje 50-100 reči.")
        
        with st.spinner("Analiziram vest..."):
            # Preprocessing
            cleaned = clean_text(news_text)
            
            # Transform
            text_vectorized = vectorizer.transform([cleaned])
            
            # Predikcija
            prediction = model.predict(text_vectorized)[0]
            prediction_proba = model.predict_proba(text_vectorized)[0]
            
            # Debug
            st.write(f"**Debug:** Prediction = {prediction} (0=FAKE, 1=TRUE)")
            st.write(f"**Debug:** Probabilities = FAKE: {prediction_proba[0]:.4f}, TRUE: {prediction_proba[1]:.4f}")
            
            # Rezultati
            st.markdown("---")
            st.subheader("📊 Rezultat Analize:")
            
            col1, col2, col3 = st.columns(3)
            
            if prediction == 1:
                # TRUE NEWS
                confidence = prediction_proba[1] * 100
                with col1:
                    st.success("✅ TRUE NEWS")
                    st.metric("Confidence", f"{confidence:.2f}%")
                with col2:
                    st.info(f"FAKE: {prediction_proba[0]*100:.2f}%")
                    st.info(f"TRUE: {prediction_proba[1]*100:.2f}%")
                with col3:
                    st.markdown("### 🎯")
                    st.markdown("**Vest je verovatno istinita**")
                
                st.success(f"Model je {confidence:.1f}% siguran da je vest ISTINITA.")
            else:
                # FAKE NEWS
                confidence = prediction_proba[0] * 100
                with col1:
                    st.error("❌ FAKE NEWS")
                    st.metric("Confidence", f"{confidence:.2f}%")
                with col2:
                    st.info(f"FAKE: {prediction_proba[0]*100:.2f}%")
                    st.info(f"TRUE: {prediction_proba[1]*100:.2f}%")
                with col3:
                    st.markdown("### ⚠️")
                    st.markdown("**Vest je verovatno lažna**")
                
                st.error(f"Model je {confidence:.1f}% siguran da je vest LAŽNA.")
            
            # Prikaži očišćen tekst
            with st.expander("🔍 Prikaži preprocesiran tekst"):
                st.text(cleaned[:500] + "..." if len(cleaned) > 500 else cleaned)

# Sidebar
st.sidebar.title("ℹ️ O Projektu")

# ✅ Prikaži info o modelu iz metadate
if metadata.get('source') == 'shared':
    st.sidebar.success("🔄 Povezano sa Pipeline-om!")
    
    trained_date = metadata.get('trained_at', 'N/A')
    if trained_date != 'N/A':
        try:
            dt = datetime.fromisoformat(trained_date)
            trained_date = dt.strftime("%d.%m.%Y %H:%M")
        except:
            trained_date = trained_date[:19]
    
    st.sidebar.markdown(f"""
    **Model Info:**
    - 🎯 Version: {metadata.get('version', 'N/A')}
    - 📊 Accuracy: {metadata.get('accuracy', 0)*100:.2f}%
    - 📅 Trained: {trained_date}
    - 📈 Dataset: {metadata.get('num_samples', 'N/A'):,} vesti
      - FAKE: {metadata.get('num_fake', 'N/A'):,}
      - TRUE: {metadata.get('num_true', 'N/A'):,}
    """)
else:
    st.sidebar.info("📦 Using local model")

st.sidebar.markdown("---")

st.sidebar.markdown("""
**Fake News Detector**

- 🧠 Model: TF-IDF + Logistic Regression
- 📚 Original Dataset: 44,898 vesti
- 🎯 Klase: FAKE / TRUE

**Kako radi:**
1. Unesi tekst vesti (engleski)
2. Klikni "Analiziraj"
3. Dobij rezultat!

**Ograničenja:**
- Model radi najbolje na dugim vestima (100+ reči)
- Kratke vesti mogu biti netačno klasifikovane
- Dataset je iz 2016-2017 perioda

**Povezan sistem:**
- 🔄 Pipeline automatski ažurira model
- 📊 Nove vesti se dodaju periodično
- 🎯 Model se retrenira sa svežim podacima
""")

st.sidebar.markdown("---")
st.sidebar.info("🎓 Projekat iz Inteligentnih Sistema")

# Footer
st.markdown("---")
st.markdown("*Developed with ❤️ using Scikit-learn & Streamlit*")