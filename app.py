import streamlit as st
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Učitaj model, tokenizer, i config
@st.cache_resource
def load_model():
    # Učitaj LSTM model (pickle)
    with open('lstm_model_complete.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Učitaj tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Učitaj config
    with open('config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    MAX_LENGTH = config['MAX_LENGTH']
    
    return model, tokenizer, MAX_LENGTH

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("🕵️ Fake News Detector")
st.markdown("### Detektuj lažne vesti pomoću AI neuronske mreže!")
st.markdown("---")

# Učitaj model
try:
    model, tokenizer, MAX_LENGTH = load_model()
    st.success("✅ Model uspešno učitan! (Bidirectional LSTM - 99.84% accuracy)")
except Exception as e:
    st.error(f"❌ Greška: {e}")
    st.info("Proveri da li su fajlovi 'lstm_model_complete.pkl', 'tokenizer.pkl', i 'config.pkl' u istom folderu!")
    st.stop()

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
    Scientists discover that drinking coffee cures all types of cancer! 
    This amazing breakthrough is being hidden by big pharma because they 
    don't want you to know the truth!
```
    
    **TRUE vest primer:**
```
    The Federal Reserve announced Wednesday that it will maintain interest 
    rates at current levels, citing concerns about inflation and economic 
    uncertainty. The decision was widely expected by economists and market analysts.
```
    """)

# Dugme za analizu
if st.button("🔍 Analiziraj Vest", type="primary"):
    if news_text.strip() == "":
        st.warning("⚠️ Molim te unesi tekst vesti!")
    else:
        with st.spinner("Analiziram vest..."):
            # Preprocessing
            cleaned = clean_text(news_text)
            sequence = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
            
            # Predikcija
            prediction = model.predict(padded, verbose=0)[0][0]
            
            # Debug
            st.write(f"**Debug:** Raw score = {prediction:.6f} (threshold = 0.5)")
            
            # Rezultati
            st.markdown("---")
            st.subheader("📊 Rezultat Analize:")
            
            col1, col2, col3 = st.columns(3)
            
            if prediction > 0.5:
                # TRUE NEWS
                confidence = prediction * 100
                with col1:
                    st.success("✅ TRUE NEWS")
                    st.metric("Confidence", f"{confidence:.2f}%")
                with col2:
                    st.info(f"Score: {prediction:.4f}")
                with col3:
                    st.markdown("### 🎯")
                    st.markdown("**Vest je verovatno istinita**")
                
                st.success(f"Model je {confidence:.1f}% siguran da je vest ISTINITA.")
            else:
                # FAKE NEWS
                confidence = (1 - prediction) * 100
                with col1:
                    st.error("❌ FAKE NEWS")
                    st.metric("Confidence", f"{confidence:.2f}%")
                with col2:
                    st.info(f"Score: {prediction:.4f}")
                with col3:
                    st.markdown("### ⚠️")
                    st.markdown("**Vest je verovatno lažna**")
                
                st.error(f"Model je {confidence:.1f}% siguran da je vest LAŽNA.")
            
            # Prikaži očišćen tekst
            with st.expander("🔍 Prikaži preprocesiran tekst"):
                st.text(cleaned[:500] + "..." if len(cleaned) > 500 else cleaned)

# Sidebar
st.sidebar.title("ℹ️ O Projektu")
st.sidebar.markdown("""
**Fake News Detector**

- 🧠 Model: Bidirectional LSTM
- 📊 Accuracy: 99.84%
- 📚 Dataset: 44,898 vesti
- 🎯 Klase: FAKE / TRUE

**Arhitektura:**
- Embedding Layer (128 dim)
- 2x Bidirectional LSTM
- Dense Layers + Dropout
- Sigmoid Output

**Kako radi:**
1. Unesi tekst vesti (engleski)
2. Klikni "Analiziraj"
3. Dobij rezultat!

**Napomena:** 
Model je treniran na engleskim vestima 
iz 2016-2017 perioda.
""")

st.sidebar.markdown("---")
st.sidebar.info("🎓 Projekat iz Inteligentnih Sistema")

# Footer
st.markdown("---")
st.markdown("*Developed with ❤️ using TensorFlow & Streamlit*")