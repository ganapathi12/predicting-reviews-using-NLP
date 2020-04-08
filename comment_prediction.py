# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ' ,dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =' '.join(review)
    corpus.append(review)
    
#creating the bag of model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
###############################################################################################
###############################################################################################
#guessing the result
def preprocess_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
        
    # here we are assuming that PortStemmer has been instantiated as ps
    review = [ps.stem(w) for w in review if w not in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review
         
def encode_review(review):
    review = preprocess_review(review)
    #print(review)        
    # here we are assuming that cv is an instance of CountVectorizer
    review = cv.transform([review]).toarray()
    return review

review = input("Enter review : ")
result = classifier.predict(encode_review(review))
print(f'review is {"positive" if result[0] == 1 else "negative"}')