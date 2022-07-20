# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
import sys
import string
import re
sys.path.append('../')
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
def train_model():
    df = pd.read_csv('model/data/selected_features_text_final.csv')
    df = df.dropna()

    X = df['text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('Vectorizing the dataset!')
    tf = TfidfVectorizer(ngram_range= (1,2), max_features = 500000)
    X_train = tf.fit_transform(X_train)
    X_test = tf.transform(X_test)
    # Time : ~
    # lr = LogisticRegression()
    # lr.fit(X_train, y_train)
    # y_pred = lr.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # conf_matrix = confusion_matrix(y_test, y_pred)
    
    # pickle.dump(lr, open('../models/79_57_logreg.pkl', 'wb'))
    print("Training")
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    pickle.dump(mnb, open('model/models/mnb_from_py.pkl', 'wb'))
    y_pred_mnb = mnb.predict(X_test)
    acc = accuracy_score(y_test, y_pred_mnb)
    print("Accuracy is", acc)

#




