# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

def clean_description(description):
    # Clean job post text
    corpus = []
    #Replace special chars by space
    desc = re.sub('[^a-zA-Z]', ' ', description)
    #To lowercase and split words
    desc = desc.lower()
    desc = desc.split()
    #Clean stopwords
    ps = PorterStemmer()
    desc = [ps.stem(word) for word in desc if not word in set(stopwords.words('english'))]
    desc = ' '.join(desc)
    corpus.append(desc)
    return corpus

def create_words_bag(corpus):
    # Creating the Bag of Words model
    # Load the CV 
    cv = joblib.load('cv.pkl')
    X = cv.transform(corpus).toarray()
    return X

# Job description to predict 
description = "Hello! Come work with me."
 
# Load model
classifier = joblib.load('fake_job.pkl') 
# Clean description
corpus = clean_description(description)
# Create word bag
X = create_words_bag(corpus)

# Predict 
pred = classifier.predict(X)
