import streamlit as st
import re
import pandas as pd
from gensim import similarities
import nltk
import gensim
from gensim import corpora
from gensim import models

# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
###############################################################################################################################################
# Import data

def load_reviews():
    df = pd.read_csv(r'C:\Users\Tina\OneDrive - Singapore Management University\SMU\Term 3\Text analytics\Project\Document retrieval\document_retrieval_update.csv')
    return df

df = load_reviews()

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

def stem_text(text):
    stemmer = PorterStemmer()
    words = text.split()  # Split the text into words
    stemmed_words = [stemmer.stem(word) for word in words]  # Apply stemming
    return ' '.join(stemmed_words)

df['review stem'] = df['Full review'].apply(stem_text)
stemmed_texts = df['review stem'].str.split()  # Split the stemmed text into lists of words
sgDictionary = corpora.Dictionary(stemmed_texts)
sgVectors = [sgDictionary.doc2bow(doc) for doc in stemmed_texts]
sgIndex = similarities.SparseMatrixSimilarity(sgVectors, len(sgDictionary))
sgTFIDF = models.TfidfModel(sgVectors)
sgVectorsWithTFIDF = [sgTFIDF[vec] for vec in sgVectors]
sgIndexWithTFIDF = similarities.SparseMatrixSimilarity(sgVectorsWithTFIDF, len(sgDictionary))

def query(text):
    qList = text.split()  
    qLower = [w.lower() for w in qList]
    stemmer = PorterStemmer()
    qStemmed = [stemmer.stem(w) for w in qLower]
    qVector = sgDictionary.doc2bow(qStemmed)
    qVectorTFIDF = sgTFIDF[qVector]
    return qVectorTFIDF, qVector

if apply_button:
    input_text = key_words if key_words != '' else input_keywords

    if input_text:
        qVectorTFIDF, qVector = query(input_text)
        simTFIDF = sgIndexWithTFIDF[qVectorTFIDF]

        df['similarity_tfidf'] = simTFIDF
        df_idf = df.sort_values(by='similarity_tfidf', ascending=False)

        pd.set_option('display.max_colwidth', None)
        columns = ['Brand', 'Review Model', 'Review rating', 'Verified Purchase or not', 
                   'list price',  'Original title', 'Original review']
        df_select = df_idf[columns]
        df_final = df_select[(df_select['Brand'] == brand) & (df_select['Review rating'] == rating)].head(4).reset_index(drop=True)
        df_final = df_final.drop_duplicates()
        df_final.index += 1
        st.dataframe(df_final)
