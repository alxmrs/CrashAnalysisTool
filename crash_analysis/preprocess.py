import functools
import re

import nltk
from nltk import pos_tag, SnowballStemmer

from crash_analysis.dataframe_helper import fill_empty


def preprocess(df, _map=None):
    """
    Preprocesses the working dataframe by tokenizing and stemming every word
    :param df: dataframe to pre-process
    :param _map: function to apply to every row of dataframe
    :return: processed dataframe
    """
    assert not isinstance(df, type(None))

    if not _map:
        _map = tokenize_stem_stop

    # remove or fill rows that have no data, i.e NaN
    nonempty_df = fill_empty(df)

    # Map each remaining row to stemmed tokens
    processed_df = nonempty_df.apply(_map)

    return processed_df


def strip_proper_pos(text):
    text = __join_if_list(text)
    try:
        tagged = pos_tag(text.split())
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
        tagged = pos_tag(text.split())

    without_propernouns = [word for word, pos in tagged if pos is not 'NPP' and pos is not 'NNPS']
    return without_propernouns


def tokenize_and_stop(text):
    text = __join_if_list(text)

    tokens = lower_and_tokenize(text)

    # exclude stopwords (like "a", "the", "in", etc.)
    stopwords = __get_stopwords()
    final_tokens = [t for t in tokens if t not in stopwords]

    return final_tokens


def tokenize_stem_stop(text):
    """
    Function that maps string input to a list of tokens. The token list has no stopwords and all words "stemmed",
    or transformed to their root word.
    :param text: Input string
    :return: list of processed tokens
    """
    text = __join_if_list(text)

    tokens = lower_and_tokenize(text)

    # stem words (e.g {installing, installed, ...} ==> install)
    # exclude stopwords (like "a", "the", "in", etc.)
    stopwords = __get_stopwords()
    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(t) for t in tokens if t not in stopwords]

    return stems


def tokenize(text, stem=True, stop=True):
    """Function that maps string input to a list of tokens. The token list has no stopwords and all words "stemmed",
    or transformed to their root word.
    :param text: Input string
    :return: list of processed tokens
    """
    def identity(x): return x
    def allow_all(x): return True

    text = __join_if_list(text)

    tokens = lower_and_tokenize(text)

    # exclude stopwords (like "a", "the", "in", etc.)
    stopwords = __get_stopwords()
    def in_stopwords(x): return x in stopwords

    # stem words (e.g {installing, installed, ...} ==> install)
    stemmer = SnowballStemmer('english')

    fin_tokens = __map_and_filter(tokens,
                                  stemmer.stem if stem else identity,
                                  in_stopwords if stop else allow_all)

    return fin_tokens



def lower_and_tokenize(text):
    # tokenize text, split by non-word characters, i.e. characters not in [a-zA-Z0-9_]
    tokens = re.split(r'\W+', text)

    # force text to lowercase, filtering out empty characters
    lower_tokens = [t.lower() for t in tokens if t != '']

    return lower_tokens


def ngram(_input, N, skip=None, delim=' ', skip_delim='_'):
    """
    ngram-ify a list of tokens.
    :param _input: input list of tokens (list of strings or a string)
    :param N: length of grams (the 'n' in ngrams)
    :param delim: Delimiter, must be a string
    :param skip_delim: Delimiter for skipped words
    :return: expanded token list of ngrams
    """
    max_len = len(_input)
    # Delimiter must be a string
    # assert isinstance(delim, basestring)
    # assert isinstance(skip_delim, basestring)

    ngram_tokens = []

    for start in range(max_len):
        for n in range(1, min(N+1, max_len+1)):
            end = start + n
            if end > len(_input):
                break
            ngram_tokens.append(delim.join(_input[start:end]))

            if skip:
                skipped = skip_delim.join(_input[start:end:(skip+1)])
                if skipped not in ngram_tokens:
                    ngram_tokens.append(skipped)

    return ngram_tokens


def skipgram(_input, N, skip=None, delim=' ', skip_delim='_'):
    """
    ngram-ify a list of tokens.
    :param _input: input list of tokens (list of strings or a string)
    :param N: length of grams (the 'n' in ngrams)
    :param delim: Delimiter, must be a string
    :param skip_delim: Delimiter for skipped words
    :return: expanded token list of ngrams
    """
    max_len = len(_input)
    # Delimiter must be a string
    # assert isinstance(delim, basestring)
    # assert isinstance(skip_delim, basestring)

    ngram_tokens = []

    for start in range(max_len):
        for n in range(1, min(N+1, max_len+1)):
            end = start + n
            if end > len(_input):
                break
            ngram_tokens.append(delim.join(_input[start:end]))

            if skip:

                for s in range(1, skip+1):

                    skipped = skip_delim.join(_input[start:end:s])
                    if skipped not in ngram_tokens:
                        ngram_tokens.append(skipped)

    return ngram_tokens



def __get_stopwords():
    """Get english stopwords from Natural Language Toolkit"""
    try:
        stopwords = nltk.corpus.stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')

    return stopwords


def __join_if_list(text_or_list):
    """Takes a string or list of strings, returns a string"""
    if isinstance(text_or_list, list):
        return ' '.join(text_or_list)
    return text_or_list


def __map_and_filter(_input, _map=lambda x: x, _filter=lambda x: True):
    """Combine map and filter into one step"""
    return [_map(x) for x in _input if _filter(x)]


def compose(*functions):
    """
    Composes a list of functions.

    For example, compose(f, g, h)(x) ==> f(g(h(x))). Works like pipe.
    :param functions: one or more functions as arguments
    :return: composition of functions.
    """
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
