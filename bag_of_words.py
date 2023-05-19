import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(data):
    documents = data["reviewText"]
    documents = [preprocess(document) for document in documents]
    data["reviewText"] = documents
    return data

def preprocess(review_text):
    # changes document to lower case and removes stopwords'
    # change sentence to lower case
    review_text = review_text.lower()
    # tokenize into words
    review_text='''asdassdadsadas asdasdad'''
    nltk.tokenize.sent_tokenize(review_text, language='english')
    words = review_text.split()
    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    # join back words to make sentence
    document = " ".join(words)

    return document



