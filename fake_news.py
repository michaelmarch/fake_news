from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

from nltk.corpus import stopwords

import numpy as np
import pandas as pd

import nltk
import string


def process_text(text):
    """Removes punctuations and stopwords from text"""
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean_wrods = [word for word in nopunc.split() if word.lower()
                   not in stopwords.words('english')]

    return clean_wrods


df_fake_news = pd.read_csv('data/fake.csv')
df_fake_news = df_fake_news.assign(label=0)

df_true_news = pd.read_csv('data/true.csv')
df_true_news = df_true_news.assign(label=1)

df_news = pd.concat([df_fake_news, df_true_news], ignore_index=True)

df_news.drop_duplicates(inplace=True)
df_news.dropna(axis=0, inplace=True)

df_news = df_news.sample(frac=1).reset_index(drop=True)

df_news['combined'] = df_news['subject'] + ' ' + \
    df_news['title'] + ' ' + df_news['date']

# Stopwords is a dictionary of words that are meaningless in data science
nltk.download('stopwords')

# Count words in each input
message_bag_of_words = CountVectorizer(
    analyzer=process_text).fit_transform(df_news['combined'])

# Split the data into 80% training set and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(
    message_bag_of_words, df_news['label'], test_size=0.9, random_state=1)

# Train the data on a classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make a prediction on the testing data and check the results (accuracy, ...)
pred = classifier.predict(X_test)
print(classification_report(y_test, pred))
