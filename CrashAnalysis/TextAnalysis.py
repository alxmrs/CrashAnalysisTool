import pandas as pd
import nltk
import sklearn
import re

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib


class TextAnalysis:
    def __init__(self):
        self.df = None
        self.working_df = None

    def read_csv(self, csv_file_path):
        """
        Reads a CSV file, converts to a pandas dataframe object.
        :param csv_file_path: text path to csv file to read
        :return: None
        :side_effects: Stores local dataframe to object
        """
        self.df = pd.read_csv(csv_file_path, low_memory=False)

    def set_dataframe(self, dataframe):
        """
        Import a dataframe to object.
        :param dataframe:
        :return: None
        :side_effects: Stores local dataframe to object
        """
        self.df = dataframe

    def filter_dataframe(self, **kwargs):
        # TODO
        """
        Generalized method to select the desired row(s) and column(s) from the dataframe. TO BE IMPLEMENTED
        :param kwargs:
        :return:
        """
        pass

    def get_customer_descriptions_by_version(self, version=None):
        """
        A temporary method to get the customer descriptions by a specific version. If no version string is specified,
        this method will return all the customer description.
        :param version: (optional) specify release version. If not specified, will default to every version
        :return: None
        :side_effects: Set working dataframe to customer data
        """
        if version:
            self.working_df = self.df['Customer_Description'][self.df.Version == version]
        else:
            self.working_df = self.df['Customer_Description']

    def preprocess(self):
        """
        Preprocesses the working dataframe by tokenizing and stemming every word
        :return:
        :side_effects: The working dataframe is lowercase, stemmed, tokenized, and has no stop words
        """
        if not self.__is_df_set(self.working_df):
            raise NotImplementedError(
                'Working dataframe not yet created! Try calling get_customer_descriptions_by_version.')

        # remove rows that have no data, i.e NaN
        self.working_df = self.__remove_empty()

        # Map each remaining row to stemmed tokens
        self.working_df = self.working_df.apply(self.__tokenize_and_stem)

    def create_vocab_frame(self):
        total_vocab_stemmed = []
        total_vocab_tokens = []

        nonempty_df = self.__remove_empty(self.working_df)

        for entry in nonempty_df:
            all_stemmed = self.__tokenize_and_stem(entry)
            total_vocab_stemmed.extend(all_stemmed)

            all_tokens = self.__tokenize_only(entry)
            total_vocab_tokens.extend(all_tokens)

        vocab_frame = pd.DataFrame({'words': total_vocab_tokens}, index=total_vocab_stemmed)

        return vocab_frame

    def frequency_count(self, data=None):
        """
        Calculates frequency count of every preprocessed word in corpus
        :return: dictionary mapping word to frequency
        """
        freq_count = {}

        if not data:
            data = self.working_df

        for entry in data:
            for word in entry:
                if word in freq_count:
                    freq_count[word] += 1
                else:
                    freq_count[word] = 1

        return freq_count

    def __tokenize_and_stem(self, text):
        """
        Function that maps string input to a list of tokens. The token list has no stopwords and all words "stemmed",
        or transformed to their root word.
        :param text: Input string
        :return: list of processed tokens
        """
        # force text to lowercase, remove beginning and trailing whitespace
        lower_text = text.lower().strip()

        # tokenize text, split by non-word characters, i.e. characters not in [a-zA-Z0-9_]
        tokens = re.split(r'\W+', lower_text)

        # stem words (e.g {installing, installed, ...} ==> install), exclude stopwords (like "a", "the", "in", etc.)
        stopwords = nltk.corpus.stopwords.words('english')
        stemmer = SnowballStemmer('english')
        stems = [stemmer.stem(t) for t in tokens if t not in stopwords]

        return stems

    def __tokenize_only(self, text):
        # force text to lowercase, remove beginning and trailing whitespace
        lower_text = text.lower().strip()

        # tokenize text, split by non-word characters, i.e. characters not in [a-zA-Z0-9_]
        tokens = re.split(r'\W+', lower_text)

        # exclude stopwords (like "a", "the", "in", etc.)
        stopwords = nltk.corpus.stopwords.words('english')
        final_tokens = [t for t in tokens if t not in stopwords]

        return final_tokens

    def __remove_empty(self, df=None):
        active_df = df if self.__is_df_set(df) else self.working_df
        return active_df.dropna(how='any')

    def __fill_empty(self, df=None):
        active_df = df if self.__is_df_set(df) else self.working_df
        return active_df.fillna(value=' ')

    def vectorize_corpus(self):
        self.working_df = self.__remove_empty()

        # create term-frequency inverse-document frequency vectorizer object
        tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                           stop_words='english',
                                           min_df=1, max_df=1.0,
                                           use_idf=True, tokenizer=self.__tokenize_and_stem, ngram_range=(1, 4))

        # transfer dataframe representation to safe list representation
        corpus = []
        for row in self.working_df:
            safe_row = re.sub(r'[^\x00-\x7F]+', ' ', row)
            corpus.append(safe_row)

        # create tf-idf matrix
        tfidf_mx = tfidf_vectorizer.fit_transform(corpus)
        terms = tfidf_vectorizer.get_feature_names()

        print('tfidf matrix shape: ')
        print(tfidf_mx.shape)

        return tfidf_mx, terms

    def kmeans(self, tfidf_mx, k=5):
        km = KMeans(n_clusters=k)

        km.fit(tfidf_mx)

        print('saving to doc cluster file...')
        joblib.dump(km, 'doc_cluster_k{0}.pkl'.format(k))

        return km

    def label_dataframe_with_clusters(self, clusters):
        df_cluster = self.df.copy()
        df_cluster = self.__remove_empty(df_cluster)
        df_cluster['Cluster'] = clusters

        df_cluster.set_index('Cluster')

        return df_cluster

    def __repr__(self):
        """
        Convert either the working dataframe (if defined) or starting dataframe to a string. Used for debugging.
        :return: string version of dataframe
        """
        active_df = self.working_df if self.__is_df_set(self.working_df) else self.df
        return str(active_df)

    def __is_df_set(self, df):
        """
        Test to see if variable is None or has a dataframe object
        :param df: dataframe to test
        :return: boolean
        """
        try:
            if not df:
                return False
        except ValueError:
            return True

    def __clear_worker(self):
        self.working_df = None

    def get_columns(self, df=None):
        """
        Returns list of column titles for input or object dataframe.
        :param df: (optional) dataframe
        :return: list of strings -- titles of dataframe columns
        :side_effects: None
        """
        active_df = df or self.df
        column_strings = [c for c in active_df.columns]
        return column_strings
