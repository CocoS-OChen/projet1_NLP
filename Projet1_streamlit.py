import streamlit as st
import os
import random
import nltk
from rank_bm25 import BM25Okapi
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet


# Assurez-vous que les ressources NLTK nécessaires sont téléchargées
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialise le lemmatiseur et l'ensemble des stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Fonction pour obtenir la nature grammaticale d'un mot pour la lemmatisation
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Fonction pour un prétraitement avancé du texte
def advanced_preprocess(text):
    word_tokens = word_tokenize(text.lower())
    filtered_words = [
        lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_tokens 
        if w not in stop_words and len(w) > 2
    ]
    return filtered_words


@st.cache_data

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


# Fonction pour charger les requêtes de l'ensemble de données NFCorpus
def load_queries(dicReq):
    return list(dicReq.values())

# Fonction pour sélectionner une requête de manière aléatoire
def get_random_query(queries):
    return random.choice(queries)

# Charge les données NFCorpus et initialise BM25
dicDoc, dicReq, dicReqDoc = loadNFCorpus()
corpus = [advanced_preprocess(doc) for doc in dicDoc.values()]
bm25 = BM25Okapi(corpus)

# Interface Streamlit pour la recherche d'informations médicales
st.title("Recherche d'Informations Médicales")
st.write("Entrez votre requête ou tirez-en une au hasard pour rechercher dans le NFCorpus.")

# Champ de saisie pour la requête
user_query = st.text_input("Posez votre question", "")

# Recherche la requête saisie par l'utilisateur
if st.button("Rechercher") and user_query:
    preprocessed_query = advanced_preprocess(user_query)
    # Récupère les scores de la requête traitée par BM25
    doc_scores = bm25.get_scores(preprocessed_query)
    # Trie les scores et récupère les indices des documents
    top_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
    # Affiche les documents les plus pertinents
    for idx in top_doc_indices[:5]:  # Affiche les 5 premiers résultats
        st.write(f"Document ID: {idx}, Score: {doc_scores[idx]}")
        st.write(dicDoc[idx])
        st.write("-----")

# Bouton pour obtenir une requête aléatoire de NFCorpus
if st.button('Question aléatoire'):
    random_query = get_random_query(load_queries(dicReq))
    st.write('Question aléatoire sélectionnée :', random_query)