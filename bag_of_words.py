import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re


def bag_of_words(data):
    # nltk.download('stopwords')
    documents = data["reviewText"]
    data = data.drop("reviewText", axis=1)
    documents = [preprocess(document) for document in documents]

    vectorizer = CountVectorizer()
    bow_model = vectorizer.fit_transform(documents)
    reviews_vector = df = pd.DataFrame(bow_model.toarray())
    data = data.join(reviews_vector)

    return data

def preprocess(review_text):
    # changes document to lower case and removes stopwords'
    # change sentence to lower case
    review_text = str(review_text).lower()
    # tokenize into words
    # nltk.tokenize.sent_tokenize(review_text, language='english')
    string_no_punctuation = re.sub("[^\w\s]", "", review_text)
    words = string_no_punctuation.split()
    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    # keep only positive or words
    # words = [word for word in words if word in positive_and_negative]
    # join back words to make sentence
    document = " ".join(words)

    return document



