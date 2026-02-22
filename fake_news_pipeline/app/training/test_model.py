import joblib

model = joblib.load("models/fake_news_model_updated.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


texts = [
   
    """
    WASHINGTON (Reuters) - The United States government announced on Monday 
    a comprehensive new education reform policy aimed at improving STEM education 
    in public schools across the nation. The Department of Education released 
    a detailed 50-page report outlining the key components of the initiative, 
    which includes increased funding for science and technology programs, 
    teacher training workshops, and new curriculum standards. Education 
    Secretary Maria Johnson stated during a press conference that the reforms 
    are expected to take effect in the fall semester of next year. The policy 
    has received bipartisan support in Congress, with lawmakers from both parties 
    praising the initiative as a necessary step toward modernizing American education.
    """,
    
 
    """
    BREAKING NEWS: Hollywood celebrity caught living secret double life as 
    alien ambassador! Shocking photos reveal the truth they don't want you 
    to see! Sources close to the celebrity claim that late-night meetings 
    with extraterrestrial beings have been happening for years. The government 
    is trying to cover this up but brave whistleblowers are coming forward 
    with the truth. Share this before it gets deleted! You won't believe 
    what happens next! Click here to see the shocking evidence that will 
    change everything you thought you knew about Hollywood!
    """,
    
   
    """
    URGENT: Aliens have officially taken control of the White House according 
    to multiple anonymous sources who claim to have inside information about 
    what really goes on behind closed doors. Secret documents allegedly show 
    proof of extraterrestrial involvement in government decisions dating back 
    decades. Mainstream media refuses to cover this story because they're 
    part of the cover-up! Wake up people! The truth is out there and they 
    don't want you to know! Share before this gets censored!
    """,
    
   
    """
    Boston, MA - Researchers at Harvard Medical School announced today the 
    publication of their latest study in the New England Journal of Medicine, 
    detailing promising developments in cancer treatment research. The study, 
    which took five years to complete and involved over 500 participants, 
    examined the efficacy of a new immunotherapy approach. Dr. Sarah Chen, 
    lead researcher on the project, cautioned that while the results are 
    encouraging, further clinical trials are needed before the treatment 
    can be approved for widespread use. The research was funded by the 
    National Institutes of Health and conducted in collaboration with 
    multiple international institutions.
    """
]


texts_tfidf = vectorizer.transform(texts)


pred = model.predict(texts_tfidf)
proba = model.predict_proba(texts_tfidf)


print("DETALJNI REZULTATI:")


for i, (text, p, prob) in enumerate(zip(texts, pred, proba)):
    label = "FAKE" if p == 0 else "TRUE"
    confidence = max(prob) * 100
    
    print(f"\n{'='*80}")
    print(f"Vest {i+1}: {label} ({confidence:.2f}% confidence)")
    print(f"{'='*80}")
    print(f"Text preview: {text[:150].strip()}...")
    print(f"Probabilities: FAKE={prob[0]:.4f} | TRUE={prob[1]:.4f}")
    
    
    expected = ["TRUE", "FAKE", "FAKE", "TRUE"][i]
    result = "✅ CORRECT" if label == expected else "❌ WRONG"
    print(f"Expected: {expected} | Result: {result}")

