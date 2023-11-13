
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Ensure NLTK resources are downloaded
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(word):
    """Function to get the part of speech tag for lemmatization."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def advanced_preprocess(text):
    word_tokens = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_tokens if w not in stop_words and len(w) > 2]
    return filtered_words

# Revising the code with the suggested optimizations

# Updated code with optimizations
#optimized_code = """
import os
import streamlit as st
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize stopwords set once
stopwords_set = set(stopwords.words('english'))

# Auxiliary functions
@st.cache_data(allow_output_mutation=True)

def loadNFCorpus():
    dir_path = "./project1-2023/"
    
    # Load documents
    filename = os.path.join(dir_path, "dev.docs")
    dicDoc = {}
    with open(filename, encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        key, value = line.split('	')
        dicDoc[key] = value

    # Load queries
    filename = os.path.join(dir_path, "dev.all.queries")
    dicReq = {}
    with open(filename, encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        key, value = line.split('	')
        dicReq[key] = value

    # Load relevance scores
    filename = os.path.join(dir_path, "dev.2-1-0.qrel")
    dicReqDoc = defaultdict(dict)
    with open(filename, encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        req, _, doc, score = line.strip().split('	')
        dicReqDoc[req][doc] = int(score)

    return dicDoc, dicReq, dicReqDoc

def text2TokenList(text):
    word_tokens = word_tokenize(text.lower())
    lemmatized_tokens = [
        lemmatizer.lemmatize(word) for word in word_tokens 
        if word not in stopwords_set and len(word) > 2
    ]
    return lemmatized_tokens

# Load NFCorpus data and initialize BM25
dicDoc, dicReq, dicReqDoc = loadNFCorpus()
corpus = [text2TokenList(doc) for doc in dicDoc.values()]
bm25 = BM25Okapi(corpus)

# Streamlit interface
st.title("Recherche d'Informations Médicales")
st.write("Entrez votre requête pour rechercher dans le NFCorpus.")

# Input field for the query
user_query = st.text_input("Requête", "")

# Search button
if st.button("Rechercher"):
    preprocessed_query = text2TokenList(user_query)
    # More code to handle the search and display results goes here...


# Returning the optimized code for further instructions or modifications
#optimized_code[:2000]  # Displaying first 2000 characters for review
