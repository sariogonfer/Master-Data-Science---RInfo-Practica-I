from time import time
import os
import pprint
import re

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import *
from sklearn.metrics.cluster import adjusted_rand_score
import nltk
import numpy
import spacy

from utils.parser import html2txt_parser_dir

nlp = spacy.load('en')

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


def timer_decorator(func):
    def func_wrapper(*args):
        start = time()
        res = func(*args)
        end = time()
        print("Tiempo total de ejecución {:.2f}".format((start - end) / 100))
        return res
    return func_wrapper


cases = list()


def register_case(func):
    cases.append(func)
    def func_wrapper(*args):
        return func(*args)
    return func_wrapper

@timer_decorator
def evaluate(func, corpus_dir="./corpus_text"):
    texts = []
    for path in sorted([f for f in os.listdir(corpus_dir)
                        if f.endswith(".txt")]):
        with open(os.path.join(corpus_dir, path), "r") as f_:
            tokens = func(f_)
            texts.append(nltk.Text(tokens))
    test = cluster_texts(texts, 5, "cosine")
    print("Clusters obtenidos: ", test)
    print("Clusters esperados: ", REFERENCE)
    return adjusted_rand_score(REFERENCE, test)


def bulk_evaluate(*funcs):
    print("Reference: " , ", ".join([str(r) for r in REFERENCE]))
    for func in funcs:
        print("Evaluando %s" % func.__name__)
        print("Puntuacion: ", evaluate(func))


def _word_tokenize(in_):
    s = in_ if isinstance(in_, str) else in_.read()
    return nltk.word_tokenize(s)


def _lemmatizer(text, expanded_stopwords=[], pos=[]):
    return [w.lemma_ for w in nlp(text) if (not (w.is_stop or str(w) in \
                expanded_stopwords or w.is_punct)) and \
                (w.pos_ in pos)]


def _to_lower_case(tokens):
    return [t.lower() for t in tokens]


def _remove_puntuation(tokens):
    import string

    return [t for t in tokens if t not in string.punctuation]


def _remove_stop_words(tokens, langs = ['english']):
    return [t for t in tokens if t not in nltk.corpus.stopwords.words(langs)]


def _remove_expanded_stop_words(tokens):
    expanded_stopwords =  nltk.corpus.stopwords.words('english') + ['the',
            'say', '-PRON-', '', 'people', 'year', 'take','international',
            'state', 'new', 'try', 'report', 'leader','government', 'tell',
            'minister', 'leave','support', 'region', 'work', 'want', 'call',
            'continue','in', 'time', 'week', 'member', 'need', 'policy',
            'news','country', 'later', 'receive', 'force', 'face', 'public',
            'sign']
    return [t for t in tokens if t not in expanded_stopwords]


def _translate_text(f_, method='textblob', target='en'):
    from textblob import TextBlob

    s = f_.read()
    tb = TextBlob(s)
    if TextBlob(s).detect_language() == target:
        return s
    if method == 'textblob':
        return str(tb.translate(to=target))
    else:
        return _translate_deepl(s)


def _get_named_entities(text, languague='en'):
    import spacy

    nlp = spacy.load('en')
    return nlp(text).ents


def _translate_deepl(s, target='EN'):
    import deepl

    return '\n'.join([deepl.translate(ss, target=target)[0]
                      for ss in s.split('.')])


@register_case
def case_1(f_):
    """ Ninguna transformación. """

    return _word_tokenize(f_)


@register_case
def case_2(f_):
    """ Convierte los tokens a minusculas. """

    tokens = _word_tokenize(f_)
    return _to_lower_case(tokens)


@register_case
def case_3(f_):
    """ Convierte a minusculas y elimina signos de puntuacion. """

    tokens = _word_tokenize(f_)
    tokens = _to_lower_case(tokens)
    return _remove_puntuation(tokens)


@register_case
def case_4(f_):
    """ Lo anterior y elimina las stopwords es español e ingles. """

    tokens = _word_tokenize(f_)
    tokens = _to_lower_case(tokens)
    tokens = _remove_puntuation(tokens)
    return _remove_stop_words(tokens, langs=['english', 'spanish'])


@register_case
def case_5(f_):
    """ Traducido con TextBlob. """

    text = _translate_text(f_)
    tokens = _word_tokenize(text)
    tokens = _to_lower_case(tokens)
    tokens = _remove_puntuation(tokens)
    return _remove_stop_words(tokens)


@register_case
def case_6(f_):
    """ Traducido con deepl. """

    text = _translate_text(f_, method='deepl')
    tokens = _word_tokenize(text)
    tokens = _to_lower_case(tokens)
    tokens = _remove_puntuation(tokens)
    return _remove_stop_words(tokens)


@register_case
def case_7(f_):
    """ Stopwords ampliadas para el corpus. """

    text = _translate_text(f_, method='deepl')
    tokens = _word_tokenize(text)
    tokens = _to_lower_case(tokens)
    tokens = _remove_puntuation(tokens)
    return _remove_expanded_stop_words(tokens)


@register_case
def case_8(f_):
    """ Con entidades nombradas. """

    text = _translate_text(f_)
    return [str(w) for w in _get_named_entities(text)]

@register_case
def case_9(f_):
    """ Con entidades nombradas filtradas """

    text = _translate_text(f_)
    return [str(w) for w in _get_named_entities(text) if w.label_ \
            in ['GPE', 'PERSON', 'NORP', 'ORG']]


@register_case
def case_10(f_):
    """ Con entidades nombradas filtradas y mas comunes. """
    from collections import Counter

    expanded_stopwords =  ['the',
            'say', '-PRON-', '', 'people', 'year', 'take','international',
            'state', 'new', 'try', 'report', 'leader','government', 'tell',
            'minister', 'leave','support', 'region', 'work', 'want', 'call',
            'continue','in', 'time', 'week', 'member', 'need', 'policy',
            'news','country', 'later', 'receive', 'force', 'face', 'public',
            'sign']


    text = _translate_text(f_)
    tokens = _lemmatizer(text, expanded_stopwords=expanded_stopwords,
                         pos=['NOUN'])
    most_common_nouns = [c[0] for c in Counter(tokens).most_common(5)]
    return _to_lower_case([str(w) for w in _get_named_entities(text) \
            if w.label_ in ['GPE', 'PERSON', 'NORP', 'ORG', 'DATE']] + \
            most_common_nouns)


def print_cases():
    print()
    for i, func in enumerate(cases):
        print('{} .- {}'.format(i, func.__doc__))

if __name__ == "__main__":
    while True:
        print_cases()
        opt = input("Seleciona una opción, (a) para todos o (q) para salir: ")
        if opt.lower() == 'q':
            break
        print('Score: ', evaluate(cases[int(opt)]))
