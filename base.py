import os
import pprint
import re

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import *
from sklearn.metrics.cluster import adjusted_rand_score
import nltk
import numpy

from utils.parser import html2txt_parser_dir


REFERENCE = [0, 5, 0, 0, 0, 2, 2, 2, 3, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 0, 2, 5]

# Function to create a TF vector for one document. For each of
# our unique words, we have a feature which is the tf for that word
# in the current document
def TF(document, unique_terms, collection):
    word_tf = []
    for word in unique_terms:
        word_tf.append(collection.tf(word, document))
    return word_tf

def cluster_texts(texts, cluster_number, distance):
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Creando collecion de %d terminos" % len(collection))

    #get a list of unique terms
    unique_terms = list(set(collection))
    print("Terminos unicos encontrados: ", len(unique_terms))

    ### And here we actually call the function and create our array of vectors.
    vectors = [numpy.array(TF(f,unique_terms, collection)) for f in texts]

    # initialize the clusterer
    clusterer = AgglomerativeClustering(n_clusters=cluster_number,
                                      linkage="average", affinity='cosine')
    clusters = clusterer.fit_predict(vectors)

    return clusters


def evaluate(func, corpus_dir="./corpus_text"):
    texts = []
    for path in sorted([f for f in os.listdir(corpus_dir)
                        if f.endswith(".txt")]):
        with open(os.path.join(corpus_dir, path), "r") as f_:
            tokens = func(f_)
            texts.append(nltk.Text(tokens))
    test = cluster_texts(texts, 5, "cosine")
    return adjusted_rand_score(REFERENCE, test)

def bulk_evaluate(*funcs):
    print("Reference: " , ", ".join([str(r) for r in REFERENCE]))
    for func in funcs:
        print("Evaluando %s" % func.__name__)
        print("Puntuacion: ", evaluate(func))

def case_1(f_):
    return nltk.word_tokenize(f_.read())

def case_2(f_):
    return [w for w in case_1(f_)
            if w not in nltk.corpus.stopwords.words(['spanish', 'english'])]

if __name__ == "__main__":
    bulk_evaluate(case_1, case_2)
