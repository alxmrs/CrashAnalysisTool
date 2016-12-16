import pandas as pd
import numpy as np
import nltk
import os
import re
import functools

from nltk.stem.snowball import SnowballStemmer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from gensim import corpora, models


# TODO [ ] convert to pure functions
# TODO [ ] comment every function

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

   def filter_dataframe(self, **kwargs):
      # TODO implement
      """
      Generalized method to select the desired row(s) and column(s) from the dataframe. TO BE IMPLEMENTED
      :param kwargs:
      :return:
      """
      pass

   def frequency(self, version=None, product_id=None, print_output=True, top=30):
      customer_desc_df = self.get_customer_descriptions(version=version, product_id=product_id)
      vocab = self.create_vocab_frame(customer_desc_df)
      processed_df = self.preprocess(customer_desc_df)

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

      for word, count in sortedFreq:

         adjacent_terms = set([val[0] for val in vocab.ix[word].get_values()])

         # intital df
         term_df = df[df['Customer_Description'].str.contains(word, case=False, na=False)]

         # concat adjacent terms to initial df
         for adj in adjacent_terms:
            term_df = term_df.add(df[df['Customer_Description'].str.contains(adj, case=False, na=False)])

         field_map[word] = term_df[field].value_counts()

         if count < min_count:
            break

      return field_map




   def lda(self, version=None, product_id=None, num_topics=5, recompute=False, multicore=True):

      # create cache model name
      cache_model_name = 'lda_model_t{0}'.format(num_topics)

      if version:
         cache_model_name += '_v{0}'.format(version)

      if product_id:
         cache_model_name += '_pid{0}'.format(''.join(re.split(r'\W', product_id)))

      cache_model_name += '.pkl'

      # load existing model
      if not recompute and os.path.isfile('./' + cache_model_name):
         lda = models.LdaModel.load(cache_model_name, mmap='r')

      # (re)compute model
      else:
         working_df = self.get_customer_descriptions(version, product_id=product_id)

         preprocessed_df = self.preprocess(working_df, compose(self.__strip_proper_POS, self.__tokenize_and_stem))

         dictionary = corpora.Dictionary(preprocessed_df)
         dictionary.filter_extremes(no_below=2, no_above=0.8)

         corpus = [dictionary.doc2bow(text) for text in preprocessed_df]

         if multicore:
            lda = models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary,
                                      chunksize=10000, passes=1000)
         else:

            lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, update_every=5,
                                  chunksize=10000, passes=1000)

         print('saving model as ' + cache_model_name)
         lda.save(cache_model_name)

      return lda

   def print_topics(self, lda_model, num_words=5):
      topics_matrix = lda_model.show_topics(formatted=False, num_words=num_words)
      # topics_matrix = np.array(topics_matrix)

      for topic in topics_matrix:
         print ('topic ' + str(topic[0]))
         print(', '.join([word_tuple[0] + ' : ' + str(word_tuple[1]) for word_tuple in topic[1]]))

   def get_customer_descriptions(self, version=None, product_id=None):
      """
      A temporary method to get the customer descriptions by a specific version. If no version string is specified,
      this method will return all the customer description.
      :param version: (optional) specify release version. If not specified, will default to every version
      :return: customer data dataframe
      """
      filtered_df = self.filter_crash_df(self.df, version, product_id)

      working_df = filtered_df['Customer_Description']

      return working_df

   def filter_crash_df(self, df, version=None, product_id=None):
      working_df = df.copy()

      if version:
         working_df = working_df[working_df.Version == version]

      if product_id:
         working_df = working_df[working_df.Product == product_id]

      return working_df

   def preprocess(self, working_df, map=None):
      """
      Preprocesses the working dataframe by tokenizing and stemming every word
      :param working_df: dataframe to pre-process
      :return: The working dataframe is lowercase, stemmed, tokenized, and has no stop words
      """
      if not self.__is_df_set(working_df):
         raise ValueError(
            'Working dataframe not yet created! Try calling get_customer_descriptions_by_version.')

      if not map:
         map = self.__tokenize_and_stem

      # remove or fill rows that have no data, i.e NaN
      nonempty_df = self.__fill_empty(working_df)

      # Map each remaining row to stemmed tokens
      processed_df = nonempty_df.apply(map)

      return processed_df

   def create_vocab_frame(self, working_df):
      total_vocab_stemmed = []
      total_vocab_tokens = []

      nonempty_df = self.__remove_empty(working_df)

      for entry in nonempty_df:
         all_stemmed = self.__tokenize_and_stem(entry)
         total_vocab_stemmed.extend(all_stemmed)

         all_tokens = self.__tokenize_only(entry)
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

      if not self.__is_df_set(data):
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

   def __tokenize_and_stem(self, text):
      """
      Function that maps string input to a list of tokens. The token list has no stopwords and all words "stemmed",
      or transformed to their root word.
      :param text: Input string
      :return: list of processed tokens
      """
      text = self.__join_if_list(text)

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

   def __tokenize_only(self, text):
      text = self.__join_if_list(text)

      # force text to lowercase, remove beginning and trailing whitespace
      lower_text = text.lower().strip()

      # tokenize text, split by non-word characters, i.e. characters not in [a-zA-Z0-9_]
      tokens = re.split(r'\W+', lower_text)

      # exclude stopwords (like "a", "the", "in", etc.)
      try:
         stopwords = nltk.corpus.stopwords.words('english')
      except LookupError as e:
         nltk.download(u'stopwords')
         stopwords = nltk.corpus.stopwords.words('english')

      final_tokens = [t for t in tokens if t not in stopwords]

      return final_tokens

   def __strip_proper_POS(self, text):
      text = self.__join_if_list(text)
      try:
         tagged = pos_tag(text.split())
      except LookupError:
         nltk.download('averaged_perceptron_tagger')
         tagged = pos_tag(text.split())

      without_propernouns = [word for word, pos in tagged if pos is not 'NPP' and pos is not 'NNPS']
      return without_propernouns

   def __join_if_list(self, text_or_list):
      if type(text_or_list) is type(list()):
         return ' '.join(text_or_list)
      return text_or_list

   def __remove_empty(self, df=None):
      if not self.__is_df_set(df):
         raise ValueError('Dataframe cannot be None.')
      return df.dropna(how='any')

   def __fill_empty(self, df=None):
      if not self.__is_df_set(df):
         raise ValueError('Dataframe cannot be None.')
      return df.fillna(value=' ')

   def vectorize_corpus(self, working_df):
      nonempty_df = working_df = self.__remove_empty(working_df)

      # create term-frequency inverse-document frequency vectorizer object
      tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                         stop_words='english',
                                         min_df=1, max_df=1.0,
                                         use_idf=True, tokenizer=self.__tokenize_and_stem, ngram_range=(1, 4))

      # transfer dataframe representation to safe list representation
      corpus = []
      for row in nonempty_df:
         safe_row = re.sub(r'[^\x00-\x7F]+', ' ', row)
         corpus.append(safe_row)

      # create tf-idf matrix
      tfidf_mx = tfidf_vectorizer.fit_transform(corpus)
      terms = tfidf_vectorizer.get_feature_names()

      print('tfidf matrix shape: ')
      print(tfidf_mx.shape)

      return tfidf_mx, terms

   def train_or_load_kmeans(self, tfidf_mx, k=5, recompute=False):

      cache_model_name = 'doc_cluster_k{0}.pkl'.format(k)

      if os.path.isfile('./' + cache_model_name) and not recompute:
         km = joblib.load(cache_model_name)
      else:
         km = KMeans(n_clusters=k)
         km.fit(tfidf_mx)
         print('saving to doc cluster file...')
         joblib.dump(km, cache_model_name)

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
      return type(df) is not type(None)

   def __clear_worker(self):
      self.working_df = None

   def get_columns(self, df=None):
      """
      Returns list of column titles for input or object dataframe.
      :param df: (optional) dataframe
      :return: list of strings -- titles of dataframe columns
      :side_effects: None
      """
      active_df = df if self.__is_df_set(df) else self.df
      column_strings = [c for c in active_df.columns]
      return column_strings


def compose(*functions):
   return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
