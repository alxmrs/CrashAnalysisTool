import functools
import re
import nltk
from nltk import pos_tag, SnowballStemmer
from typing import List, Sequence, TypeVar, Callable, Optional, Union, Any
from crash_analysis.dataframe_helper import fill_empty
import pandas as pd

T = TypeVar('T')
K = TypeVar('K')


def preprocess(df: pd.DataFrame, _map: Optional[Callable[[T], Any]] = None) -> pd.DataFrame:
    """Preprocesses the working dataframe by tokenizing and stemming every word

    This function is can be overloaded with any type of "_map", or function to apply to every row of the dataframe. 
    By default, it will use tokenize_stem_stop. 
    
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


def strip_proper_pos(text: Union[List[str], str]) -> List[str]:
    """Return list of words as long as they are not proper nouns
    
    Works via part-of-speech (pos) tagging from the natural language tool kit (nltk). 
    
    :param text: 
    :return: list of non-pronoun words. 
    """

    text = __join_if_list(text)
    try:
        tagged = pos_tag(text.split())
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
        tagged = pos_tag(text.split())

    without_propernouns = [word for word, pos in tagged if pos is not 'NPP' and pos is not 'NNPS']
    return without_propernouns


def tokenize_and_stop(text: Union[List[str], str]) -> List[str]:
    """Lowers, tokenizes input and removes stop words. 
    
    :param text: A list of tokens or a string
    :return: List of processed tokens
    
    >>> tokenize('I Love Coding, Running, And Eating', stem=False)
    ['love', 'coding', 'running', 'eating']
    """

    text = __join_if_list(text)

    tokens = lower_and_tokenize(text)

    # exclude stopwords (like "a", "the", "in", etc.)
    stopwords = __get_stopwords()
    final_tokens = [t for t in tokens if t not in stopwords]

    return final_tokens


def tokenize_stem_stop(text: Union[List[str], str]):
    """Tokenizes, stems, and removes stop words. 
    
    Function that maps string input to a list of tokens. The token list has no stopwords and all words "stemmed",
    or transformed to their root word.
    :param text: Input string
    :return: list of processed tokens
    
    >>> tokenize_stem_stop('I Love Coding, Running, And Eating')
    ['love', 'code', 'run', 'eat']
    """
    text = __join_if_list(text)

    tokens = lower_and_tokenize(text)

    # stem words (e.g {installing, installed, ...} ==> install)
    # exclude stopwords (like "a", "the", "in", etc.)
    stopwords = __get_stopwords()
    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(t) for t in tokens if t not in stopwords]

    return stems


def tokenize(text: Union[List[str], str], stem: bool = True, stop: bool = True) -> List[str]:
    """Generalized tokenization function. 
    
    Function that maps string input to a list of tokens. The token list has no stopwords and all words "stemmed",
    or transformed to their root word.
    :param text: Input string
    :param stem: should you stem the tokens?
    :param stop: should you remove stop words?
    :return: list of processed tokens
    
    >>> tokenize('I Love Coding, Running, And Eating')
    ['love', 'code', 'run', 'eat']
    >>> tokenize('I Love Coding, Running, And Eating', stop=False)
    ['i', 'love', 'code', 'run', 'and', 'eat']
    >>> tokenize('I Love Coding, Running, And Eating', stem=False)
    ['love', 'coding', 'running', 'eating']
    """
    def identity(x): return x

    def allow_all(x): return True

    text = __join_if_list(text)

    tokens = lower_and_tokenize(text)

    # exclude stopwords (like "a", "the", "in", etc.)
    stopwords = __get_stopwords()

    def not_stopword(x): return not x in stopwords

    # stem words (e.g {installing, installed, ...} ==> install)
    stemmer = SnowballStemmer('english')

    fin_tokens = __map_and_filter(tokens,
                                  stemmer.stem if stem else identity,
                                  not_stopword if stop else allow_all)

    return fin_tokens


def lower_and_tokenize(text: str) -> List[str]:
    """lowers input, converts string to list of tokens (delimited by whitespace).
    
    :param text: 
    :return: List of processed tokens
    
    >>> lower_and_tokenize('I Love Coding, Running, And Eating')
    ['i', 'love', 'coding', 'running', 'and', 'eating']
    """

    # tokenize text, split by non-word characters, i.e. characters not in [a-zA-Z0-9_]
    tokens = re.split(r'\W+', text)

    # force text to lowercase, filtering out empty characters
    lower_tokens = [t.lower() for t in tokens if t != '']

    return lower_tokens


def ngram(_input: List[str], N: int, skip: Optional[int] = None,
          delim: str = ' ', skip_delim: str = '_') -> List[str]:
    """
    ngram-ify a list of tokens.
    
    ngrams, to summarize, are groups of single words and n-tuples of words. 
    
    See https://en.wikipedia.org/wiki/N-gram for reference. 
    
    :param _input: input list of tokens (list of strings or a string)
    :param N:  length of grams (the 'n' in ngrams)
    :param skip: Number of tokens to skip
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
            if end > max_len:
                break
            ngram_tokens.append(delim.join(_input[start:end]))

            if skip:
                skipped = skip_delim.join(_input[start:end:(skip+1)])
                # if skipped not in ngram_tokens:
                ngram_tokens.append(skipped)

    return ngram_tokens


def skipgram(_input: List[str], N: int, skip: Optional[int] = None,
             delim: str = ' ', skip_delim: str = '_') -> List[str]:
    """A variation on ngrams that allows "skips" between tokens. 
    
    Will add all skips from 1 to N. 
    
    :param _input: input list of tokens (list of strings or a string)
    :param N: length of grams (the 'n' in ngrams)
    :param skip: Number of tokens to skip
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
                    # if skipped not in ngram_tokens:
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


def __join_if_list(text_or_list: Union[List[str], str]) -> str:
    """Takes a string or list of strings, returns a string
    
    :param text_or_list: either a string or list of tokens
    :return: a string
    
    >>> __join_if_list(['the', 'cat', 'in', 'the', 'hat'])
    'the cat in the hat'
    >>> __join_if_list('the cat in the hat')
    'the cat in the hat'
    """

    if isinstance(text_or_list, list):
        return ' '.join(text_or_list)
    return text_or_list


def __map_and_filter(_input: Sequence[T],
                     _map: Callable[[T], Any] = lambda x: x,
                     _filter: Callable[[T], bool] = lambda x: True) -> Sequence[Any]:
    """Combine map and filter into one step
    
    :param _input: list of data
    :param _map: function to map all elements of data (default: identity)
    :param _filter: function to filter all elements of data (default, include all)
    :return: mapped and filtered list of data
    
    >>> __map_and_filter(range(1,11))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> __map_and_filter(range(1,11), lambda m: m**2, lambda f: f % 2 == 0)
    [4, 16, 36, 64, 100]
    
    """

    return [_map(x) for x in _input if _filter(x)]


def compose(*functions):
    """Composes a list of functions.

    For example, compose(f, g, h)(x) ==> f(g(h(x))). Works like pipe.
    :param functions: one or more functions as arguments
    :return: composition of functions.
    
    >>> compose(lambda x: x + 1, lambda y: y*2, lambda z: z**2)(2)
    9
    >>> compose(print, int)("10")
    10
    """
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


if __name__ == '__main__':
    import doctest

    doctest.testmod()