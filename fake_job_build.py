# NLP classifier to classify fake job posts based on job post description 
# By: Rami D
# Creation Date: 14-03-20

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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('fake_job_postings.csv')

# The dataset is unbalancde we must resample it after spliting
dataset['fraudulent'].value_counts().plot(kind='bar')

# Drop null values
dataset.dropna(inplace=True, subset=['description', 'fraudulent'])

# Resampling the dataset to balance the class

# separate minority and majority classes
negative = dataset[dataset['fraudulent'] == 0]
positive = dataset[dataset['fraudulent'] == 1]

# upsample minority
pos_upsampled = resample(negative,replace=True, n_samples=len(positive), random_state=27)
# combine majority and upsampled minority
upsampled = pd.concat([positive, pos_upsampled])

# After resampling
upsampled['fraudulent'].value_counts().plot(kind='bar')

# Handle description 
jobs_description = upsampled.loc[: ,'description'].values

# Cleaning the jobs description text
corpus = []
for description in jobs_description:
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

# Creating the Bag of Words model
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = upsampled.iloc[:, 17].values

# Save vectorizer 
joblib.dump(cv, 'cv.pkl') 


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluate the model 
cm = confusion_matrix(y_test, y_pred)

# Save the model
joblib.dump(classifier, 'fake_job.pkl') 

