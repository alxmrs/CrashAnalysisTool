from __future__ import print_function

import os
import re


from sklearn.cluster import KMeans
from sklearn.externals import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

from crash_analysis.preprocess import tokenize_stem_stop
from crash_analysis import remove_empty


def vectorize_corpus(working_df):
    nonempty_df = working_df = remove_empty(working_df)

    # create term-frequency inverse-document frequency vectorizer object
    tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                       stop_words='english',
                                       min_df=1, max_df=1.0,
                                       use_idf=True, tokenizer=tokenize_stem_stop, ngram_range=(1, 4))

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


def top_terms_per_cluster(frame, km, num_clusters, vocab_frame, terms):
    print("Top terms per cluster:")
    print()

    # sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        print("Cluster %d words:" % i, end='')

        for ind in order_centroids[i, :10]:  # replace 6 with n words per cluster
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'),
                  end=',')
        print()
        print()

        # print("Cluster %d titles:" % i, end='')
        # for err_code in frame.ix[i]['Error_Code'].values.tolist():
        #     print(' %s,' % err_code, end='')
        # print()
        # print()

    print()
    print()

# mx, terms = tool.vectorize_corpus()
#
#
#
# compute = True
# n_custers = 7
# if compute:
#     km = tool.kmeans(mx, n_custers)
# else:
#     km = joblib.load('doc_cluster_k{0}.pkl'.format(n_custers))
#
# clusters = km.labels_.tolist()
# cluster_lists = [[x] for x in clusters]
#
# print(tool.frequency_count(cluster_lists))
#
# new_df = tool.label_dataframe_with_clusters(clusters)
#
# print(new_df['Cluster'].value_counts())
#
# # tool.label_frame_with_clusters(clusters)
#
# top_terms_per_cluster(new_df, km, n_custers, vocab_frame, terms)

