import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re


def bag_of_words(data, positive_and_negative):
    # nltk.download('stopwords')
    documents = data["reviewText"]
    data = data.drop("reviewText", axis=1)
    documents = [preprocess(document, positive_and_negative) for document in documents]

    vectorizer = CountVectorizer()
    bow_model = vectorizer.fit_transform(documents)
    reviews_vector = df = pd.DataFrame(bow_model.toarray())
    data = data.join(reviews_vector)

    return data


def preprocess(review_text, positive_and_negative):
    # change sentence to lower case
    review_text = str(review_text).lower()
    # tokenize into words
    # nltk.tokenize.sent_tokenize(review_text, language='english')
    string_no_punctuation = re.sub("[^\w\s]", "", review_text)
    words = string_no_punctuation.split()
    # remove stop words
    # words = [word for word in words if word not in stopwords.words("english")]
    # keep only positive or words
    words = [word for word in words if word in positive_and_negative["word1"].values]
    # join back words to make sentence
    document = " ".join(words)

    return document


# TODO clean this up
def preprocess_without_joining_words(review_text, positive_and_negative):
    # change sentence to lower case
    review_text = str(review_text).lower()
    # tokenize into words
    # nltk.tokenize.sent_tokenize(review_text, language='english')
    string_no_punctuation = re.sub("[^\w\s]", "", review_text)
    words = string_no_punctuation.split()
    # remove stop words
    # words = [word for word in words if word not in stopwords.words("english")]
    # keep only positive or words
    words = [word for word in words if word in positive_and_negative["word1"].values]

    return words



def count_pos_and_neg(review_text, positive_and_negative):
    # df = pd.DataFrame("n_of_pos", "n_of_neg", "n_of_both", "n_of_neutral")
    # df["n_of_pos"] = []
    # df["n_of_neg"] = []
    # df["n_of_both"] = []
    # df["n_of_neutral"] = []

    numbers = [0, 0, 0, 0]

    for word in review_text:
        row = positive_and_negative.loc[positive_and_negative['word1'] == word]
        print(row.columns)
        positivity = row.at[0, 'priorpolarity']
        if positivity == "positive":
            numbers[0] += 1

    return numbers


def vectorize_with_lexicon(data, positive_and_negative):
    documents = data["reviewText"]
    data = data.drop("reviewText", axis=1)
    documents = [preprocess_without_joining_words(document, positive_and_negative) for document in documents]
    reviews_vector = [count_pos_and_neg(document, positive_and_negative) for document in documents]
    data = data.join(reviews_vector)

    return data
