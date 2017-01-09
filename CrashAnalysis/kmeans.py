import os
import re

from sklearn.cluster import KMeans
from sklearn.externals import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

from CrashAnalysis.preprocess import remove_empty, tokenize_and_stem


def vectorize_corpus(working_df):
    nonempty_df = working_df = remove_empty(working_df)

    # create term-frequency inverse-document frequency vectorizer object
    tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                       stop_words='english',
                                       min_df=1, max_df=1.0,
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 4))

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


def train_or_load_kmeans(tfidf_mx, k=5, recompute=False):

    cache_model_name = 'doc_cluster_k{0}.pkl'.format(k)

    if os.path.isfile('./' + cache_model_name) and not recompute:
        km = joblib.load(cache_model_name)
    else:
        km = KMeans(n_clusters=k)
        km.fit(tfidf_mx)
        print('saving to doc cluster file...')
        joblib.dump(km, cache_model_name)

    return km