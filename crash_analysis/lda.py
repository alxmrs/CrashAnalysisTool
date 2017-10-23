import os
import re

from gensim import models, corpora


"""
The methods in this file should not be regularly used for crash analysis. 
After investigation, I determined that the problem descriptions from customers are not large enough 
for effective document clustering or topic models.  
"""

def lda(preprocessed_df, version=None, product_id=None, num_topics=5, recompute=False, multicore=True):

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
        # working_df = self.get_customer_descriptions(version, product_id=product_id)
        # preprocessed_df = self.preprocess(working_df, compose(strip_proper_POS, tokenize_and_stem))

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


def print_topics(lda_model, num_words=5):
    topics_matrix = lda_model.show_topics(formatted=False, num_words=num_words)
    # topics_matrix = np.array(topics_matrix)

    for topic in topics_matrix:
        print('topic ' + str(topic[0]))
        print(', '.join([word_tuple[0] + ' : ' + str(word_tuple[1]) for word_tuple in topic[1]]))