import nltk
from nltk.util import bigrams
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from gensim import corpora, similarities, models
import streamlit as st

# Load your data from the Excel file
df = pd.read_excel("amazon_review_processed_full.xlsx")

# Define a function to preprocess and tokenize the text
def preprocess_text(row):
    # Combine the title and review columns and convert to lowercase
    title = row['Original title'] if not pd.isnull(row['Original title']) else ''
    review = row['Original review'] if not pd.isnull(row['Original review']) else ''
    text = (title + ' ' + review).lower()
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenization into unigrams
    unigrams = word_tokenize(text)
    
    # Tokenization into bigrams
    bigrams_list = list(bigrams(unigrams))
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_unigrams = [stemmer.stem(word) for word in unigrams]
    stemmed_bigrams = [tuple(stemmer.stem(word) for word in bigram) for bigram in bigrams_list]
    
    # Define a set of stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    unigrams_without_stopwords = [word for word in stemmed_unigrams if word not in stop_words]
    
    # Convert bigrams to strings and remove stopwords
    bigrams_without_stopwords = [' '.join(bigram) for bigram in stemmed_bigrams if not any(word in stop_words for word in bigram)]
    
    # Join both unigrams and bigrams 
    tokens = unigrams_without_stopwords + bigrams_without_stopwords
    
    return tokens

# Apply the modified preprocessing function to your DataFrame
df['Processed Text'] = df.apply(preprocess_text, axis=1)
# Apply the preprocessing function to your DataFrame
df['Processed Text'] = df.apply(preprocess_text, axis=1)

# Create a dictionary from the tokenized content
dictionary = corpora.Dictionary(df['Processed Text'])

# Create a corpus (bag of words) from the tokenized content
corpus = [dictionary.doc2bow(text) for text in df['Processed Text']]

# To find the similarity scores, create a reverse index
Index = similarities.SparseMatrixSimilarity(corpus, len(dictionary))

# Create a TFIDF reverse index
TFIDF = models.TfidfModel(corpus)
corpus_TFIDF = [TFIDF[vec] for vec in corpus]
Index_TFIDF = similarities.SparseMatrixSimilarity(corpus_TFIDF, len(dictionary))

# A function to preprocess and tokenize the query
def preprocess_query(query):
    # Convert the query to lowercase
    query = query.lower()
    
    # Remove special characters and punctuation
    query = re.sub(r'[^\w\s]', ' ', query)
    
    # Tokenization into unigrams
    unigrams = word_tokenize(query)
    
    # Tokenization into bigrams
    bigrams_list = list(bigrams(unigrams))
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_unigrams = [stemmer.stem(word) for word in unigrams]
    stemmed_bigrams = [tuple(stemmer.stem(word) for word in bigram) for bigram in bigrams_list]
    
    # Define a set of stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    unigrams_without_stopwords = [word for word in stemmed_unigrams if word not in stop_words]
    bigrams_without_stopwords = [' '.join(bigram) for bigram in stemmed_bigrams if not any(word in stop_words for word in bigram)]
    
    # Join both unigrams and bigrams 
    tokens = unigrams_without_stopwords + bigrams_without_stopwords
    
    return tokens


# A function to search and sort the results based on a query
def search_and_sort(query):
    # Import nltk inside the app logic
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    nltk.download('punkt')
    
    # Preprocess the query and obtain a list of tokens
    preprocessed_query = preprocess_query(query)

    # Create a query vector using the same dictionary as the corpus
    qVector = dictionary.doc2bow(preprocessed_query)

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
