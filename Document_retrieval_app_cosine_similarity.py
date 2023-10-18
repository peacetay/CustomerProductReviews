import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from gensim import corpora, similarities, models
import streamlit as st

# Load your data from the Excel file
df = pd.read_excel("amazon_review_processed_full.xlsx")

# Tokenization (Full Review)
df['Tokenized Full review'] = df['Full review'].apply(lambda x: word_tokenize(str(x)) if isinstance(x, str) else [])

# Create a dictionary from the tokenized content
dictionary = corpora.Dictionary(df['Tokenized Full review'])

# Create a corpus (bag of words) from the tokenized content
corpus = [dictionary.doc2bow(text) for text in df['Tokenized Full review']]

# To find the similarity scores, create a reverse index
Index = similarities.SparseMatrixSimilarity(corpus, len(dictionary))

# Create a TFIDF reverse index
TFIDF = models.TfidfModel(corpus)
corpus_TFIDF = [TFIDF[vec] for vec in corpus]
Index_TFIDF = similarities.SparseMatrixSimilarity(corpus_TFIDF, len(dictionary))

# A function to search and sort the results based on a query
def search_and_sort(query):
    # Import nltk inside the app logic
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    nltk.download('punkt')
    
    qList = query.split()
    qLower = [w.lower() for w in qList]

    # Stem it
    stemmer = PorterStemmer()
    qStemmed = [stemmer.stem(w) for w in qLower]

    # Create a query vector using the same dictionary as the corpus
    qVector = dictionary.doc2bow(qStemmed)

    # Get its TFIDF from the same model as the corpus
    qVectorTFIDF = TFIDF[qVector]

    # Get the similarities from the two indexes (raw and TFIDF)
    simRaw = Index[qVector]
    simTFIDF = Index_TFIDF[qVectorTFIDF]

    # Create a DataFrame to store the results
    results_raw = pd.DataFrame({
        'Document': range(len(df)),
        'Raw Similarity Score': simRaw
    })

    results_tfidf = pd.DataFrame({
        'Document': range(len(df)),
        'TFIDF Similarity Score': simTFIDF
    })

    # Add the Similarity Score as a new column to the original DataFrame
    df['Raw Similarity Score'] = results_raw['Raw Similarity Score']
    df['TFIDF Similarity Score'] = results_tfidf['TFIDF Similarity Score']

    # Sort the DataFrame by TFIDF Similarity Score in descending order
    df_sorted_tfidf = df.sort_values(by='TFIDF Similarity Score', ascending=False)
    
    return df_sorted_tfidf[['Review Model', 'Review date', 'Review rating', 'Original title', 'Original review', 'TFIDF Similarity Score']].head(10)

# Streamlit app
st.title("Amazon Review Search")
query = st.text_input("Enter your query:")
if st.button("Search"):
    if query:
        results = search_and_sort(query)
        st.table(results)
    else:
        st.warning("Please enter a query.")

