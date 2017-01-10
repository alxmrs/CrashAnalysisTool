import os

import pandas as pd

from CrashAnalysis.preprocess import preprocess, remove_empty, is_df_set, tokenize_and_stop, tokenize_stem_stop


class TextAnalysis:
    def __init__(self, file_or_dataframe=None):
        self.df = None
        self.working_df = None

        if type(file_or_dataframe) is type('test'):
            self.df = self.read_csv(file_or_dataframe).copy()
        elif type(file_or_dataframe) is pd.DataFrame:
            self.df = file_or_dataframe
        else:
            raise TypeError('invalid input type: please enter a string to a filepath or a dataframe')

    def read_csv(self, csv_file_path):
        """
        Reads a CSV file, converts to a pandas dataframe object.
        :param csv_file_path: text path to csv file to read
        :return: None
        :side_effects: Stores local dataframe to object
        """

        pwd = os.getcwd()
        os.chdir(os.path.dirname(csv_file_path))

        df = pd.read_csv(os.path.basename(csv_file_path), low_memory=False)

        os.chdir(pwd)
        return df

    def filter_crash_df(self, df, version=None, product_id=None):
        working_df = df.copy()

        if version:
            working_df = working_df[working_df.Version == version]

        if product_id:
            working_df = working_df[working_df.Product == product_id]

        return working_df

    def frequency(self, version=None, product_id=None, print_output=True, top=30):
        customer_desc_df = self.get_customer_descriptions(version=version, product_id=product_id)
        vocab = self.create_vocab_frame(customer_desc_df)
        processed_df = preprocess(customer_desc_df)

        freq_count, total = self.count_entries(processed_df)

        sortedFreq = []

        for w in sorted(freq_count, key=freq_count.get, reverse=True):
            sortedFreq.append((w, freq_count[w]))

        if print_output:
            print('total words: ' + str(total))
            for i in range(min(top, len(sortedFreq))):
                if type(vocab) is not type(None):
                    print('{0:10} : {1:4} \t {2}'.format(sortedFreq[i][0], sortedFreq[i][1],
                                                         vocab.ix[sortedFreq[i][0]].values.tolist()[:6]))
                else:
                    print('{0:10} : {1:4}'.format(sortedFreq[i][0], sortedFreq[i][1]))

        return vocab, sortedFreq

    def group_by_vocab(self, vocab, sortedFreq, df, field='Error_Code', min_count=0):
        field_map = dict()
        term_count_map = dict()

        for word, count in sortedFreq:

            adjacent_terms = set([val[0] for val in vocab.ix[word].get_values()])

            # # intital df
            # term_df = df[df['Customer_Description'].str.contains(word, case=False, na=False)]
            #
            # # concat adjacent terms to initial df
            # for adj in adjacent_terms:
            #    term_df = term_df.add(df[df['Customer_Description'].str.contains(adj, case=False, na=False)])

            term_df = pd.concat([df[df['Customer_Description'].fillna(' ').str.contains(term, case=False, na=False)]
                                 for term in adjacent_terms]).drop_duplicates('CrashGUID')

            num_rows_with_word = sum([term_df['Customer_Description'].str.contains(term, case=False, na=False)
                                     .value_counts()[True] for term in adjacent_terms])

            field_map[word] = term_df[field].value_counts()
            term_count_map[word] = num_rows_with_word

            if count < min_count:
                break

        return field_map, term_count_map

    def get_customer_descriptions(self, version=None, product_id=None):
        """
        TODO: depricate
        A temporary method to get the customer descriptions by a specific version. If no version string is specified,
        this method will return all the customer description.
        :param version: (optional) specify release version. If not specified, will default to every version
        :return: customer data dataframe
        """
        filtered_df = self.filter_crash_df(self.df, version, product_id)

        working_df = filtered_df['Customer_Description']

        return working_df

    def create_vocab_frame(self, working_df):
        total_vocab_stemmed = []
        total_vocab_tokens = []

        nonempty_df = remove_empty(working_df)

        for entry in nonempty_df:
            all_stemmed = tokenize_stem_stop(entry)
            total_vocab_stemmed.extend(all_stemmed)

            all_tokens = tokenize_and_stop(entry)
            total_vocab_tokens.extend(all_tokens)

        vocab_frame = pd.DataFrame({'words': total_vocab_tokens}, index=total_vocab_stemmed)

        return vocab_frame

    def count_entries(self, data=None):
        """
        Calculates frequency count of every preprocessed word in corpus. Specifically returns dictionary with counts and a
        total field, which contains the total words in the dictionary.

        :param data: dataframe or list of lists to count word frequency
        :return: tuple with dictionary mapping word to count and a total field with the total words in the dictionary
        """
        freq_count = {}
        total = 0

        if not is_df_set(data):
            # data = self.working_df
            raise ValueError('Input cannot be None.')

        for entry in data:
            for word in entry:
                if word in freq_count:
                    freq_count[word] += 1
                else:
                    freq_count[word] = 1

                total += 1

        return freq_count, total

    def label_dataframe_with_clusters(self, clusters):
        df_cluster = self.df.copy()
        df_cluster = remove_empty(df_cluster)
        df_cluster['Cluster'] = clusters

        df_cluster.set_index('Cluster')

        return df_cluster

    def __repr__(self):
        """
        Convert either the working dataframe (if defined) or starting dataframe to a string. Used for debugging.
        :return: string version of dataframe
        """
        active_df = self.working_df if is_df_set(self.working_df) else self.df
        return str(active_df)

    def __clear_worker(self):
        self.working_df = None

    def get_columns(self, df=None):
        """
        Returns list of column titles for input or object dataframe.
        :param df: (optional) dataframe
        :return: list of strings -- titles of dataframe columns
        :side_effects: None
        """
        active_df = df if is_df_set(df) else self.df
        column_strings = [c for c in active_df.columns]
        return column_strings


def filter_dataframe(df, **kwargs):
    # TODO test
    """
    Generalized method to select the desired row(s) and column(s) from the dataframe. TO BE IMPLEMENTED
    :param df:
    :param kwargs:
    :return:
    """
    working_df = df.copy()

    for key, value in kwargs.iteritems():
        value = value if isinstance(value, list) else [value]
        working_df = working_df[working_df[str(key)].isin(value)]

    return working_df
