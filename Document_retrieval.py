import streamlit as st
import re
import pandas as pd
import tensorflow_hub as hub
from nltk.corpus import stopwords
from openai.embeddings_utils import cosine_similarity
###############################################################################################################################################
# Import data

def load_reviews():
    df = pd.read_excel(r'C:\Users\Tina\OneDrive - Singapore Management University\SMU\Term 3\Text analytics\Project\amazon_review JOIN.xlsx')
    return df

df = load_reviews()

###############################################################################################################################################
# Functions and cleaning

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    return text

df['Cleaned Reviews'] = df['Review Content'].apply(clean_text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def brand(model):
    match = re.search(r'^(HP|Epson|Canon)', str(model), re.IGNORECASE)
    if match:
        return match.group(0)

def embed(text, model):
    embeddings = model([text])
    return embeddings.numpy()[0]

model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

df['Final Reviews'] = df['Cleaned Reviews'].apply(remove_stopwords)
df['Brand'] = df['Review Model'].apply(brand)
df['Embed_sentence'] = df['Final Reviews'].apply(lambda x: embed(x, model))

###############################################################################################################################################
# Header and filter

Brand = ['HP', 'Canon', 'Epson']

st.header("Select printer brand")

st.write("""Type keywords to find related reviews.""")

cols = st.columns(1)
brand = cols[0].selectbox(
    label="Brand",
    options=Brand
)

Key_words = st.text_input('Keywords')
feature_apply = st.button('Apply')

def embed_label(text, model):
    embeddings = model([text])
    return embeddings.numpy()[0]

if feature_apply:
    input_embedding_vector = embed_label(Key_words, model)
    df['similarity'] = df['Embed_sentence'].apply(lambda x: cosine_similarity(input_embedding_vector,x))

    pd.set_option('display.max_colwidth', None)
    df_new = df.sort_values(by='similarity', ascending=False)
    columns = ['Brand', 'Review Model', 'Review rating', 'Review Content']
    df_select = df_new[columns]
    df_final = df_select[df_select['Brand'] == brand].head(10).reset_index(drop=True)
    st.dataframe(df_final)
