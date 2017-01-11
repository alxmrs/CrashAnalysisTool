import pandas as pd

from crash_analysis.preprocess import preprocess, tokenize_and_stop, tokenize_stem_stop
from dataframe_helper import remove_empty, fill_empty, read_csv, get_column, filter_dataframe


class TextAnalysis:
    def __init__(self, file_or_dataframe=None):
        self.df = None
        self.working_df = None

        if type(file_or_dataframe) is type('test'):
            self.df = read_csv(file_or_dataframe).copy()
        elif type(file_or_dataframe) is pd.DataFrame:
            self.df = file_or_dataframe
        else:
            raise TypeError('invalid input type: please enter a string to a filepath or a dataframe')

    def __repr__(self):
        """
        Convert either the working dataframe (if defined) or starting dataframe to a string. Used for debugging.
        :return: string version of dataframe
        """
        active_df = self.working_df if not isinstance(self.working_df, type(None)) else self.df
        return str(active_df)

    def __clear_worker(self):
        self.working_df = None


def stem_frequency(df, column=None, _map=None, print_output=True, top=30, **filters):
    """
    Calculate the frequency of stemmed keywords in the corpus

    :param df: source dataframe
    :param column: text column of dataframe
    :param print_output: formatted printing of frequency of stems, along with example source tokens
    :param top: print only the top n items (default: 30)
    :param filters: use to filter rows out of dataframe. list of arguments in Key=Value format.
    :return: (vocab, sorted_counts, total)
        vocab: vocabulary dataframe which maps stemmed token to list of unstemmed tokens
        sorted_counts: list of tuples of stemmed_token and count, sorted by count
        total: total words in the corpus
    """

    # Get a text column from the dataframe, filtering the proper rows
    if column:
        text_df = get_column(filter_dataframe(df, filters), column)
    else:
        text_df = df

    # Match stemmed words to a list of their unstemmed tokens
    if _map:
        vocab = None
    else:
        vocab = create_vocab_frame(text_df)

    # apply default preprocessing to the text column
    processed_df = preprocess(text_df, _map=_map)

    counts, total = count_entries(processed_df)

    # sort the counts dictionary into list of tuples
    sorted_counts = []

    for w in sorted(counts, key=counts.get, reverse=True):
        sorted_counts.append((w, counts[w]))

    # optionally print formatted output
    if print_output:
        print('total words: ' + str(total))
        for i in range(min(top, len(sorted_counts))):
            if not isinstance(vocab, type(None)):
                print('{0:10} \t: {1:4} \t {2}'.format(sorted_counts[i][0],
                                                             sorted_counts[i][1],
                                                             vocab.ix[sorted_counts[i][0]].values.tolist()[:6]))
            else:
                print('{0:10} \t: {1:4}'.format(sorted_counts[i][0],
                                                       sorted_counts[i][1]))

    return sorted_counts, total, vocab


def count_entries(data):
    """
    Calculates frequency count of every preprocessed word in corpus. Specifically returns dictionary with counts and a
    total field, which contains the total words in the dictionary.

    :param data: dataframe or list of lists to count word frequency
    :return: tuple with dictionary mapping word to count and a total field with the total words in the dictionary
    """
    assert not isinstance(data, type(None))

    freq_count = {}
    total = 0

    for entry in data:
        for word in entry:
            if word in freq_count:
                freq_count[word] += 1
            else:
                freq_count[word] = 1

            total += 1

    return freq_count, total


def create_vocab_frame(text_df_column):
    """Create dataframe mapping stemmed token to list of unstemmed tokens from a column dataframe of text data"""
    total_vocab_stemmed = []
    total_vocab_tokens = []

    nonempty_df = remove_empty(text_df_column)

    for entry in nonempty_df:
        all_stemmed = tokenize_stem_stop(entry)
        total_vocab_stemmed.extend(all_stemmed)

        all_tokens = tokenize_and_stop(entry)
        total_vocab_tokens.extend(all_tokens)

    vocab_frame = pd.DataFrame({'words': total_vocab_tokens}, index=total_vocab_stemmed)

    return vocab_frame


def associate_by_keyterms(df, text_column, field='Error_Code', print_output=True, min_count=20, **filters):
    """
    Groups histograms (value_counts) of column values of a dataframe by keyterms. For example, you are able to find out
    which stack traces are most commonly associated with each keyterm.

    :param df: source dataframe
    :param text_column: column of text in the source dataframe
    :param field: which field you want to associate with the keyterm
    :param print_output: (optional, default: True) print keyterm/field associations
    :param min_count: keyterms that appear less than this threshold will not be printed
    :param filters: an argument list in Key=Value format used to filter the source dataframe
    :return: (filed_map, term_count_map)
        field_map: dictionary that maps keyterms to a value_count (histogram dataframe) of the field
        term_count_map: dictionary that maps keyterm to number of occurrence of that term in the source dataframe
    """
    field_map = dict()
    term_count_map = dict()

    text_df = get_column(filter_dataframe(df, filters), text_column)

    vocab = create_vocab_frame(text_df)
    sorted_counts, total = count_entries(text_df)

    for word, count in sorted_counts:

        # get set of unstemmed tokens from stemmed token vocabulary
        adjacent_terms = set([val[0] for val in vocab.ix[word].get_values()])

        # Concate dataframes that contain the adjacent terms into one dataframe
        adjacent_dfs = [df[fill_empty(text_df).str.contains(term, case=False, na=False)] for term in adjacent_terms]
        term_df = pd.concat(adjacent_dfs)

        num_rows_with_word = sum([term_df[text_column].str.contains(term, case=False, na=False)
                                 .value_counts()[True] for term in adjacent_terms])

        field_map[word] = term_df[field].value_counts()
        term_count_map[word] = num_rows_with_word

        if count < min_count:
            break

    if print_output:
        print('{0} by Keyterm'.format(field))
        for word, count in sorted_counts:
            print('keyterm: ' + word)
            print(field_map[word][field_map[word] > 0])

            if count < min_count:
                break

    return field_map, term_count_map