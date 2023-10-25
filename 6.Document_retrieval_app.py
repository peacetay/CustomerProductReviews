import streamlit as st
import re
import pandas as pd
from gensim import similarities
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk.corpus import stopwords
from nltk.stem.porter import *
import gensim
from gensim import corpora
from gensim import models
import joblib

# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
###############################################################################################################################################
# Import data

df = joblib.load('processed_data.joblib')

###############################################################################################################################################
# Header and filter

Brand = ['HP', 'Canon', 'Epson']

st.header("Find related reviews")
st.write("Select a topic to find related reviews.")

brand = st.selectbox("Select Brand", Brand)

labels = [
    '',
    'Setup',
    'Connectivity',
    'Customer Support',
    'Print Quality',
    'Print Speed',
    'Ink Supply and Cartridge',
    'Ease of Use',
    'Firmware',
    'Business Services and Subscription',
    'Paper Jam',
    'Control Panel'
]

Rating = [1, 2, 3, 4, 5]
key_words = st.selectbox("Select Topic", labels)
if key_words == "":
    with st.expander('Or type your own keywords'):
        input_keywords = st.text_input('Type Keywords')
else:
    with st.expander('Or type your own keywords'):
        input_keywords = ""

# with st.expander('Or type your own keywords'):
#     input_keywords = st.text_input('Type Keywords')
rating = st.selectbox("Select Rating", Rating)
apply_button = st.button('Apply')

# compute similarity score from result of TFIDF Bigram

dictionary_bi = corpora.Dictionary(df['Processed_bigram'])
corpus_bi = [dictionary_bi.doc2bow(text) for text in df['Processed_bigram']]
TFIDF_bi = models.TfidfModel(corpus_bi)
corpus_TFIDF_bi = [TFIDF_bi[vec] for vec in corpus_bi]
IndexTFIDF_bi = similarities.SparseMatrixSimilarity(corpus_TFIDF_bi, len(dictionary_bi))


# Process query
def query(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        unigrams = word_tokenize(text)
        bigrams_list = list(bigrams(unigrams))

        # Stemming
        stemmer = PorterStemmer()
        stemmed_unigrams = [stemmer.stem(word) for word in unigrams]
        stemmed_bigrams = [tuple(stemmer.stem(word) for word in bigram) for bigram in bigrams_list]

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        unigrams_without_stopwords = [word for word in stemmed_unigrams if word not in stop_words]
        bigrams_without_stopwords = [' '.join(bigram) for bigram in stemmed_bigrams if not any(word in stop_words for word in bigram)]

        # Join both unigrams and bigrams
        tokens = unigrams_without_stopwords + bigrams_without_stopwords 
        
        qVector_bi = dictionary_bi.doc2bow(tokens)
        qVectorTFIDF_bi = TFIDF_bi[qVector_bi]
        return qVectorTFIDF_bi

if apply_button:
    input_text = key_words if key_words != '' else input_keywords

    if input_text:    
        qVectorTFIDF_bi = query(input_text)
        simTFIDF_bi = IndexTFIDF_bi[qVectorTFIDF_bi]
        df['Similarity_TFIDF_Bigram'] = simTFIDF_bi
        df_tfidf_bi = df.sort_values(by = 'Similarity_TFIDF_Bigram', ascending=False)
        selected_columns = ["Review Model", "Review date", "Review rating", "Original title", "Original review", "Similarity_TFIDF_Bigram", "Brand"]
        pd.set_option('display.max_colwidth', None)
        df_select = df_tfidf_bi[selected_columns]
        df_select = df_select.reset_index(drop=True)  # Reset the index
        df_final = df_select[(df_select['Brand'] == brand) & (df_select['Review rating'] == rating)].head(5)
        df_final = df_final.drop_duplicates()
        df_final.index += 1
        st.dataframe(df_final)