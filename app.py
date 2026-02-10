import streamlit as st
import re
import joblib


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@st.cache_resource
def load_model():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('lr_model.pkl')
    return tfidf, model


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("🕵️ Fake News Detector")
st.markdown("### Detektuj lažne vesti pomoću AI modela!")
st.markdown("---")


try:
    tfidf, model = load_model()
    st.success("✅ Model uspešno učitan! (TF-IDF + Logistic Regression - 99.24% accuracy)")
except Exception as e:
    st.error(f"❌ Greška: {e}")
    st.info("Proveri da li su fajlovi 'tfidf_vectorizer.pkl' i 'lr_model.pkl' u istom folderu!")
    st.stop()


st.subheader("📝 Unesi tekst vesti:")
news_text = st.text_area(
    "Kopiraj vest ovde (mora biti na ENGLESKOM):",
    height=200,
    placeholder="Paste your news article here..."
)


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
    WASHINGTON (Reuters) - President Barack Obama signed into law today 
    a new defense spending bill. The legislation includes provisions for 
    military equipment upgrades and personnel increases. Congressional 
    leaders praised the bipartisan effort to pass the measure.
```
    """)


if st.button("🔍 Analiziraj Vest", type="primary"):
    if news_text.strip() == "":
        st.warning("⚠️ Molim te unesi tekst vesti!")
    else:
        with st.spinner("Analiziram vest..."):
           
            cleaned = clean_text(news_text)
            
            
            text_tfidf = tfidf.transform([cleaned])
            
           
            prediction = model.predict(text_tfidf)[0]
            prediction_proba = model.predict_proba(text_tfidf)[0]
            
          
            st.write(f"**Debug:** Prediction = {prediction} (0=FAKE, 1=TRUE)")
            st.write(f"**Debug:** Probabilities = FAKE: {prediction_proba[0]:.4f}, TRUE: {prediction_proba[1]:.4f}")
            
            
            st.markdown("---")
            st.subheader("📊 Rezultat Analize:")
            
            col1, col2, col3 = st.columns(3)
            
            if prediction == 1:
               
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
            
            
            with st.expander("🔍 Prikaži preprocesiran tekst"):
                st.text(cleaned[:500] + "..." if len(cleaned) > 500 else cleaned)


st.sidebar.title("ℹ️ O Projektu")
st.sidebar.markdown("""
**Fake News Detector**

- 🧠 Model: TF-IDF + Logistic Regression
- 📊 Accuracy: 99.24%
- 📚 Dataset: 44,898 vesti
- 🎯 Klase: FAKE / TRUE

**Kako radi:**
- TF-IDF ekstrakcija features (5000)
- N-grams: (1,2)
- Classifier: Logistic Regression
- Training set: 80%
- Test set: 20%

**Proces:**
1. Unesi tekst vesti (engleski)
2. Klikni "Analiziraj"
3. Dobij rezultat!

**Dataset:** 
Vesti iz perioda 2016-2017
""")

st.sidebar.markdown("---")
st.sidebar.info("🎓 Projekat iz Inteligentnih Sistema")


st.markdown("---")
st.markdown("*Developed with ❤️ using Scikit-learn & Streamlit*")