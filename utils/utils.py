from functools import lru_cache
from nltk.corpus import wordnet
import nltk
import string


@lru_cache(maxsize=32)
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def get_clean_tokens_from_str(s):
    tokens = nltk.tokenize.word_tokenize(s.lower())
    stop_words = nltk.corpus.stopwords.words('english')
    lem = nltk.stem.WordNetLemmatizer()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [lem.lemmatize(t) for t in tokens]
    return tokens

def get_tagged_tokens_from_str(s):
    tokens = get_clean_tokens_from_str(s)
    tagged_tokens = nltk.pos_tag(tokens)
    return tagged_tokens
