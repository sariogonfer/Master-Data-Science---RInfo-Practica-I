# -*- coding: utf-8 -*-

from time import time
import os
import pprint
import re

from IPython.display import display, HTML
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import *
from sklearn.metrics.cluster import adjusted_rand_score
import nltk
import numpy
import spacy
import pandas as pd

from utils.parser import html2txt_parser_dir

pd.set_option('display.max_colwidth', -1)

nlp = spacy.load('en')

REFERENCE = [0, 5, 0, 0, 0, 2, 2, 2, 3, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 0, 2, 5]
LANG_REF = ['E', 'E', 'E', 'S', 'E', 'E', 'E', 'E', 'S', 'S', 'E', 'E', 'E',
            'S', 'E', 'E', 'S', 'E', 'E', 'E', 'E', 'S']

# Function to create a TF vector for one document. For each of
# our unique words, we have a feature which is the tf for that word
# in the current document
def TF(document, unique_terms, collection):
    word_tf = []
    for word in unique_terms:
        word_tf.append(collection.tf(word, document))
    return word_tf


def TF_idf(document, unique_terms, collection):
    word_tf_idf = []
    for word in unique_terms:
        word_tf_idf.append(collection.tf_idf(word, document))
    return word_tf_idf


def cluster_texts(texts, cluster_number, distance, verbose=True, measure=TF):
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)

    #get a list of unique terms
    unique_terms = list(set(collection))

    if verbose:
        print("Creando collecion de %d terminos" % len(collection))
        print("Terminos unicos encontrados: ", len(unique_terms))

    ### And here we actually call the function and create our array of vectors.
    vectors = [numpy.array(measure(f,unique_terms, collection)) for f in texts]

    # initialize the clusterer
    clusterer = AgglomerativeClustering(n_clusters=cluster_number,
                                      linkage="average", affinity='cosine')
    clusters = clusterer.fit_predict(vectors)

    return clusters


'''
Decoradores auxiliares.
'''

def timer_decorator(func):
    def func_wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        end = time()
        diff_time = end - start
        if (diff_time < 60):
            print("Tiempo total de ejecución {:.2f} segundos".format(diff_time))
        else:
            print("Tiempo total de ejecución {:.2f} minutos".format(diff_time/60))
        return res
    return func_wrapper



cases = list()


def register_case(func):
    cases.append(func)
    def func_wrapper(*args):
        return func(*args)
    return func_wrapper


'''
Funciones auxiliares para pintar.
'''

def print_cases():
    print()
    for i, func in enumerate(cases):
        print('{} .- {}'.format(i + 1, func.__doc__))


def print_score(score):
    print("La puntuacion obtenida ha sido: ", score)


def print_clusters_table(test):
    df = pd.DataFrame.from_items([("Idiomas", LANG_REF), ("Ref.", REFERENCE),
            ("Test", test)], orient='index', columns=range(0, len(test) - 1))
    display(df)


'''
Evaluadores. Metodos usados para ejecutar los casos.
'''

def _evaluate(func, corpus_dir, verbose, measure):
    texts = []
    for path in sorted([f for f in os.listdir(corpus_dir)
                        if f.endswith(".txt")]):
        with open(os.path.join(corpus_dir, path), "r") as f_:
            tokens = func(f_)
            texts.append(nltk.Text(tokens))
    test = cluster_texts(texts, 5, "cosine", verbose, measure)
    if verbose:
        print_clusters_table(test)
    return adjusted_rand_score(REFERENCE, test)


@timer_decorator
def evaluate(func, corpus_dir="./corpus_text", measure=TF):
    return _evaluate(func, corpus_dir, True, measure)


def evaluate_all(corpus_dir="./corpus_text", measure=TF):
    scores = list()
    for case in cases:
        scores.append(_evaluate(case, corpus_dir, False, measure))

    df = pd.DataFrame(scores, columns=['Scores'],
                      index=[f.__doc__ for f in cases])
    return df


def word_tokenize(in_):
    s = in_ if isinstance(in_, str) else in_.read()
    return nltk.word_tokenize(s)


def lemmatizer(text, expanded_stopwords=[], pos=[]):
    return [w.lemma_ for w in nlp(text) if (not (w.is_stop or str(w) in \
                expanded_stopwords or w.is_punct)) and \
                (w.pos_ in pos)]


def to_lower_case(tokens):
    return [t.lower() for t in tokens]


def remove_puntuation(tokens):
    import string

    return [t for t in tokens if t not in string.punctuation]


def remove_stop_words(tokens, langs = ['english']):
    return [t for t in tokens if t not in nltk.corpus.stopwords.words(langs)]


def remove_expanded_stop_words(tokens):
    expanded_stopwords =  nltk.corpus.stopwords.words('english') + ['the',
            'say', '-PRON-', '', 'people', 'year', 'take','international',
            'state', 'new', 'try', 'report', 'leader','government', 'tell',
            'minister', 'leave','support', 'region', 'work', 'want', 'call',
            'continue','in', 'time', 'week', 'member', 'need', 'policy',
            'news','country', 'later', 'receive', 'force', 'face', 'public',
            'sign']
    return [t for t in tokens if t not in expanded_stopwords]


def translate_text(f_, method='textblob', target='en'):
    from textblob import TextBlob

    s = f_.read()
    tb = TextBlob(s)
    if TextBlob(s).detect_language() == target:
        return s
    if method == 'textblob':
        return str(tb.translate(to=target))
    else:
        return translate_deepl(s)


def get_named_entities(text, languague='en'):
    return nlp(text).ents


def translate_deepl(s, target='EN'):
    import deepl

    return '\n'.join([deepl.translate(ss, target=target)[0]
                      for ss in s.split('.')])


@register_case
def case_1(f_):
    """ Ninguna transformación. """

    return word_tokenize(f_)


@register_case
def case_2(f_):
    """ Convierte los tokens a minusculas. """

    tokens = word_tokenize(f_)
    return to_lower_case(tokens)


@register_case
def case_3(f_):
    """ Convierte a minusculas y elimina signos de puntuacion. """

    tokens = word_tokenize(f_)
    tokens = to_lower_case(tokens)
    return remove_puntuation(tokens)


@register_case
def case_4(f_):
    """ Lo anterior y elimina las stopwords es español e ingles. """

    tokens = word_tokenize(f_)
    tokens = to_lower_case(tokens)
    tokens = remove_puntuation(tokens)
    return remove_stop_words(tokens, langs=['english', 'spanish'])


@register_case
def case_5(f_):
    """ Traducido con TextBlob. """

    text = translate_text(f_)
    tokens = word_tokenize(text)
    tokens = to_lower_case(tokens)
    tokens = remove_puntuation(tokens)
    return remove_stop_words(tokens)


@register_case
def case_51(f_):
    """ Traducido con TextBlob (a español). """

    text = translate_text(f_, target='es')
    tokens = word_tokenize(text)
    tokens = to_lower_case(tokens)
    tokens = remove_puntuation(tokens)
    return remove_stop_words(tokens, langs=['spanish'])


@register_case
def case_6(f_):
    """ Traducido con deepl. """

    text = translate_text(f_, method='deepl')
    tokens = word_tokenize(text)
    tokens = to_lower_case(tokens)
    tokens = remove_puntuation(tokens)
    return remove_stop_words(tokens)


@register_case
def case_7(f_):
    """ Stopwords ampliadas para el corpus. """

    text = translate_text(f_)
    tokens = word_tokenize(text)
    tokens = to_lower_case(tokens)
    tokens = remove_puntuation(tokens)
    return remove_expanded_stop_words(tokens)


@register_case
def case_8(f_):
    """ Con entidades nombradas. """

    text = translate_text(f_)
    return [str(w).lower() for w in get_named_entities(text)]


@register_case
def case_9(f_):
    """ Con entidades nombradas filtradas """

    text = translate_text(f_)
    return [str(w).lower() for w in get_named_entities(text) if w.label_ \
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


    text = translate_text(f_)
    tokens = lemmatizer(text, expanded_stopwords=expanded_stopwords,
                         pos=['NOUN'])
    most_common_nouns = [c[0] for c in Counter(tokens).most_common(5)]
    return to_lower_case([str(w) for w in get_named_entities(text) \
            if w.label_ in ['GPE', 'PERSON', 'NORP', 'ORG', 'DATE']] + \
            most_common_nouns)

@register_case
def case_11(f_):
    """ Con entidades nombradas filtradas y mas comunes eliminando duplicados. """

    #Devolvemos
    return set(case_10(f_))


if __name__ == "__main__":
    while True:
        print_cases()
        opt = input("Seleciona una opción, (a) para todos o (q) para salir: ")
        if opt.lower() == 'q':
            break
        if opt.lower() == 'a':
            display(evaluate_all())
            continue
        if int(opt):
            print_score(evaluate(cases[int(opt) - 1]))
