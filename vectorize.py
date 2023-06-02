import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


def bag_of_words(data):
    documents = data["reviewText"]
    data = data.drop("reviewText", axis=1)
    documents = [preprocess(document) for document in documents]

    vectorizer = CountVectorizer()
    bow_model = vectorizer.fit_transform(documents)
    reviews_vector = df = pd.DataFrame(bow_model.toarray())
    data = data.join(reviews_vector)

    return data


def preprocess(review_text):
    # change sentence to lower case
    review_text = str(review_text).lower()
    # tokenize into words
    # nltk.tokenize.sent_tokenize(review_text, language='english')
    string_no_punctuation = re.sub("[^\w\s]", "", review_text)
    words = string_no_punctuation.split()
    # remove stop words and stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in stopwords.words("english")]

    # join back words to make sentence
    document = " ".join(words)

    return document


def preprocess_with_lexicon(review_text, positive_and_negative):
    # change sentence to lower case
    review_text = str(review_text).lower()
    # tokenize into words
    # nltk.tokenize.sent_tokenize(review_text, language='english')
    string_no_punctuation = re.sub("[^\w\s]", "", review_text)
    words = string_no_punctuation.split()
    # keep only positive or negative words
    words = [word for word in words if word in positive_and_negative["word1"].values]

    return words


def count_pos_and_neg(review_text, positive_and_negative):
    numbers = [0, 0, 0, 0]

    for word in review_text:
        row = positive_and_negative.loc[positive_and_negative['word1'] == word]
        positivity = row.iat[0, row.columns.get_loc('priorpolarity')]
        if positivity == "positive":
            numbers[0] += 1
        elif positivity == "negative":
            numbers[1] += 1
        elif positivity == "both":
            numbers[2] += 1
        elif positivity == "neutral":
            numbers[3] += 1

    return numbers


def vectorize_with_lexicon(data, positive_and_negative):
    documents = data["reviewText"]
    data = data.drop("reviewText", axis=1)
    documents = [preprocess_with_lexicon(document, positive_and_negative) for document in documents]
    reviews_vector = [count_pos_and_neg(document, positive_and_negative) for document in documents]
    data = data.join(pd.DataFrame(reviews_vector, columns=["n_of_pos", "n_of_neg", "n_of_both", "n_of_neutral"]))
    return data
