import functools
import re

import nltk
from nltk import pos_tag, SnowballStemmer


def preprocess(working_df, map=None):
    """
    Preprocesses the working dataframe by tokenizing and stemming every word
    :param working_df: dataframe to pre-process
    :return: The working dataframe is lowercase, stemmed, tokenized, and has no stop words
    """
    if not is_df_set(working_df):
        raise ValueError(
            'Working dataframe not yet created! Try calling get_customer_descriptions_by_version.')

    if not map:
        map = tokenize_and_stem

    # remove or fill rows that have no data, i.e NaN
    nonempty_df = fill_empty(working_df)

    # Map each remaining row to stemmed tokens
    processed_df = nonempty_df.apply(map)

    return processed_df


def remove_empty(df=None):
    if not is_df_set(df):
        raise ValueError('Dataframe cannot be None.')
    return df.dropna(how='any')


def fill_empty(df=None):
    if not is_df_set(df):
        raise ValueError('Dataframe cannot be None.')
    return df.fillna(value=' ')


def is_df_set(df):
    """
    Test to see if variable is None or has a dataframe object
    :param df: dataframe to test
    :return: boolean
    """
    # type(df) is not type(None)  # old version
    return isinstance(df, None)


def strip_proper_POS(text):
    text = join_if_list(text)
    try:
        tagged = pos_tag(text.split())
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
        tagged = pos_tag(text.split())

    without_propernouns = [word for word, pos in tagged if pos is not 'NPP' and pos is not 'NNPS']
    return without_propernouns


def tokenize_only(text):
    text = join_if_list(text)

    # force text to lowercase, remove beginning and trailing whitespace
    lower_text = text.lower().strip()

    # tokenize text, split by non-word characters, i.e. characters not in [a-zA-Z0-9_]
    tokens = re.split(r'\W+', lower_text)

    # exclude stopwords (like "a", "the", "in", etc.)
    try:
        stopwords = nltk.corpus.stopwords.words('english')
    except LookupError:
        nltk.download(u'stopwords')
        stopwords = nltk.corpus.stopwords.words('english')

    final_tokens = [t for t in tokens if t not in stopwords]

    return final_tokens


def tokenize_and_stem(text):
    """
    Function that maps string input to a list of tokens. The token list has no stopwords and all words "stemmed",
    or transformed to their root word.
    :param text: Input string
    :return: list of processed tokens
    """
    text = join_if_list(text)

    # force text to lowercase, remove beginning and trailing whitespace
    lower_text = text.lower().strip()

    # tokenize text, split by non-word characters, i.e. characters not in [a-zA-Z0-9_]
    tokens = re.split(r'\W+', lower_text)

    # stem words (e.g {installing, installed, ...} ==> install), exclude stopwords (like "a", "the", "in", etc.)
    try:
        stopwords = nltk.corpus.stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')

    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(t) for t in tokens if t not in stopwords]

    return stems


def ngram(input, N, delim=' '):
    """
    ngram-ify a list of tokens.
    :param input: input list of tokens (list of strings or a string)
    :param N: length of grams (the 'n' in ngrams)
    :param delim: Delimiter, must be a string
    :return: expanded token list of ngrams
    """
    # N cannot be greater than the length of the list
    assert N <= len(input)
    # Delimiter must be a string
    assert isinstance(delim, basestring)

    ngram_tokens = []

    for start in range(len(input)):
        for n in range(1, N+1):
            end = start + n
            if end > len(input):
                break
            ngram_tokens.append(delim.join(input[start:end]))

    return ngram_tokens


def join_if_list(text_or_list):
    if isinstance(text_or_list, list):
        return ' '.join(text_or_list)
    return text_or_list


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
