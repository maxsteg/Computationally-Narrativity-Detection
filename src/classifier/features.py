import os

seed_value = 42  # random seed of 42 for all experiments
os.environ['PYTHONHASHSEED'] = str(seed_value)

from gensim.models import LdaModel
import config
import string
import csv
import pandas as pd
import numpy as np
from nltk.util import ngrams
from string import punctuation
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.corpora import Dictionary
from lexical_diversity import lex_div as ld
from top2vec import Top2Vec

np.random.seed(seed_value)
BOOK_PATH = "../../dataset/BookNLP_output/"
print('\n----\nUsing the BookNLP path:', BOOK_PATH)


def temporal_order(fname):
    df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv', delimiter=',')
    return float(df.loc[df['FILENAME'] == fname]['temporal_order'])


def temporality(fname):
    df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv', delimiter=',')
    return float(df.loc[df['FILENAME'] == fname]['temporality'])


def saying(fname):
    df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv', delimiter=',')
    return float(df.loc[df['FILENAME'] == fname]['saying'])


def setting(fname):
    df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv', delimiter=',')
    return float(df.loc[df['FILENAME'] == fname]['setting'])


def concreteness(fname):
    """
    Returns a ratio for concreteness:
    Numerator --> sum of concreteness scores of every word/bigram in the text that appears in Brysbaert et al's lexicon.
    Denominator -->  # words (excluding punctuations)

    Parameters
    ----------
    word_list: list of words

    Returns
    -------
    concreteness score (float)
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    word_list = get_words(df)

    conc_score = 0.0
    for n in [1, 2]:
        for tup in ngrams(word_list, n):
            temp = ''  # temp is the unigram/bigram/trigram
            for word in tup:
                temp += ' ' + word.lower()
            temp = temp.strip()
            if temp in config.MAP_WORD_CONC:  # check if it exists in lexicon
                conc_score += config.MAP_WORD_CONC[temp]
    #                 print("Exists:", temp, "| Score:", config.MAP_WORD_CONC[temp])
    return conc_score / len(word_list)


def eventfulness(fname):
    df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv', delimiter=',')
    return float(df.loc[df['FILENAME'] == fname]['eventfulness'])


def feltness(fname):
    df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv', delimiter=',')
    return float(df.loc[df['FILENAME'] == fname]['feltness'])


def coherence(fname):
    df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv', delimiter=',')
    return float(df.loc[df['FILENAME'] == fname]['coh_seq'])


def agency(fname):
    df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv', delimiter=',')
    return float(df.loc[df['FILENAME'] == fname]['agency'])


def agenthood(fname):
    df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv', delimiter=',')
    return float(df.loc[df['FILENAME'] == fname]['agenthood'])


def filter_punct(tokens):
    """
    Given a list of all tokens, returns a list of words. Punctuations are skipped.
    """
    words = []
    for token in tokens:
        if str(token) not in string.punctuation:
            words.append(token)
    return words


def get_words(df):
    """
    Given the BookNLP dataframe, returns a list of words. Punctuations are skipped.
    """
    df = df.loc[df['dependency_relation'] != 'punct']  # remove punctuations
    words = filter_punct(df['word'].tolist())  # filter punctuations again in case BookNLP missed any
    return words


def get_POS_str(fname):
    """
    Returns a string for all part-of-speech tags in the given filename.
    """
    df = pd.read_csv(BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    return ' '.join(df['POS_tag'].tolist())


def get_dep_str(fname):
    """
    Returns a string for all dependency tags in the given filename.
    """
    df = pd.read_csv(BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    return ' '.join(df['dependency_relation'].tolist())


def get_words_str(fname):
    """
    Returns a string for all 'word' tokens in the given BookNLP DataFrame.
    """
    df = pd.read_csv(BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    return ' '.join(df['word'].tolist())


def filter_punct(tokens):
    """
    Removes all punctuations and returns a list of words.
    """
    return [word for word in tokens if word not in punctuation]


def compute_quoted_words(fname):
    """
    Returns fraction of number of quoted words to total number of words.
    """
    df_quotes = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.quotes', delimiter='\t', quoting=csv.QUOTE_NONE)
    df_quotes.fillna("", inplace=True)
    quoted_words = []
    for row in df_quotes['quote']:
        quoted_words += (row.split(' '))

    quoted_words = filter_punct(quoted_words)

    df_tokens = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df_tokens.fillna("", inplace=True)
    all_words = filter_punct(get_words(df_tokens))

    return len(quoted_words) / len(all_words)


def time(fname):
    """
    Returns number time entities in text
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.supersense', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    N_TIME = df.loc[df['supersense_category'] == 'noun.time'].shape[0]
    return N_TIME


def perception(fname):
    """
    Returns number of perception verbs in text
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.supersense', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    N_PERCEPTION = df.loc[df['supersense_category'] == 'verb.perception'].shape[0]
    return N_PERCEPTION


def doc2vec(train_x, test_x, preload=True):
    """
    Returns Doc2Vec vector representations of input file names

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    preload: default True, loads model pretrained on complete dataset (17k+ documents)

    Returns
    ----------
    train_vec, test_vec

    """
    train_x_words, test_x_words = [], []

    # When preload is False, train model using the train_x files
    if preload == False:
        for fname in train_x:
            df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
            df.fillna("", inplace=True)
            train_x_words.append(TaggedDocument(get_words(df), [fname]))
    for fname in test_x:
        df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
        df.fillna("", inplace=True)
        test_x_words.append(get_words(df))

    # Load pretrained model
    if preload == True:
        doc2vec_model = Doc2Vec.load('doc2vec/d2vmodel')
    else:
        doc2vec_model = Doc2Vec(train_x_words, vector_size=200, workers=1, seed=seed_value, epochs=20)

    # Infer vectors for train and test documents
    train_vec = []
    for fname in train_x:
        train_vec.append(doc2vec_model.dv[fname])
    train_vec = np.array(train_vec)

    test_vec = []
    for i in test_x_words:
        test_vec.append(doc2vec_model.infer_vector(i))
    test_vec = np.array(test_vec)

    return train_vec, test_vec


def topic_modelling(train_x, test_x, preload=True):
    """
    Returns LDA vector representations of input file names

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    preload: default True, loads model pretrained on complete dataset (17k+ documents)

    Returns
    ----------
    train_vec, test_vec

    """
    train_x_words, test_x_words = [], []
    for fname in train_x:
        df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
        df.fillna("", inplace=True)
        train_x_words.append(get_words(df))
    for fname in test_x:
        df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
        df.fillna("", inplace=True)
        test_x_words.append(get_words(df))
    train_x_dict = Dictionary(train_x_words)
    train_corpus = [train_x_dict.doc2bow(text) for text in train_x_words]

    # When preload is False, train model using the train_x files
    if preload == False:
        NUM_TOPICS = 50
        lda = LdaModel(train_corpus, num_topics=NUM_TOPICS, random_state=42)

    # Load pretrained model
    else:
        lda = LdaModel.load('lda/ldamodel')
        NUM_TOPICS = len(lda.get_topics())
    test_x_dict = Dictionary(test_x_words)
    test_corpus = [test_x_dict.doc2bow(text) for text in test_x_words]

    # Infer vectors for train and test documents
    train_vec = []
    for doc in train_corpus:
        prob_distribution = [0.0 for i in range(NUM_TOPICS)]
        distribution = lda.get_document_topics(doc)
        for i in distribution:
            prob_distribution[i[0]] = i[1]
        train_vec.append(np.array(prob_distribution))
    train_vec = np.array(train_vec)

    test_vec = []
    for doc in test_corpus:
        prob_distribution = [0.0 for i in range(NUM_TOPICS)]
        distribution = lda.get_document_topics(doc)
        for i in distribution:
            prob_distribution[i[0]] = i[1]
        test_vec.append(np.array(prob_distribution))
    test_vec = np.array(test_vec)

    return train_vec, test_vec


def improved_concreteness(fname):
    """
    Returns a ratio for concreteness:
    Numerator --> sum of concreteness scores of every word/bigram in the text that appears in Brysbaert et al's lexicon.
    Denominator -->  # words (excluding punctuations)

    Parameters
    ----------
    word_list: list of words

    Returns
    -------
    concreteness score (float)
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    word_list = get_words(df)

    conc_score = 0.0
    for n in [i for i in range(1, 30)]:  # 30 is length of the longest expression in lexicon
        for exp in ngrams(word_list, n):
            temp = ''
            for word in exp:
                temp += ' ' + word.lower()
            temp = temp.strip()
            if temp in config.EXTENDED_MAP_WORD_CONC:  # check if it exists in lexicon
                conc_score += config.EXTENDED_MAP_WORD_CONC[temp]
                # print("Exists:", temp, "| Score:", config.EXTENDED_MAP_WORD_CONC[temp])
    return conc_score / len(word_list)


def mtld_bid(fname):
    """
    Returns the bidirectional MTLD score

    Parameters
    ----------
    word_list: list of words

    Returns
    -------
    Bidirectional MTLD score (float)
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    word_list = filter_punct(get_words(df))
    return ld.mtld_ma_bid(word_list)


def ttr(fname):
    """
    Returns the TTR score

    Parameters
    ----------
    word_list: list of words

    Returns
    -------
    TTR score (float)
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    word_list = filter_punct(get_words(df))
    return ld.ttr(word_list)


def adjadv(fname):
    """
    Returns the proportion of adverbs and adjectives in a text passage:
    Numerator --> sum of number of adverbs and adjectives in the text.
    Denominator -->  # words (excluding punctuations)

    Parameters
    ----------
    N_ADJ: number adjectives
    N_ADV: number adverbs
    N_WORD: number words without punctuation

    Returns
    -------
    Proportion ADJ+ADV in document (float)
    """
    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)

    N_ADJ = df.loc[df['POS_tag'] == 'ADJ'].shape[0]
    N_ADV = df.loc[df['POS_tag'] == 'ADV'].shape[0]
    N_WORD = df.loc[df['POS_tag'] != 'PUNCT'].shape[0]

    return (N_ADJ + N_ADV) / N_WORD


def sequencers(fname):
    """
    Returns the proportion of sequencers in a text passage:
    Numerator --> sum of number of 'sequencers'
    Denominator -->  # words (excluding punctuations)

    Parameters
    ----------

    Returns
    -------
    Proportion sequencers in document (float)
    """
    sequencers = {'at last', 'furthermore', 'finally', 'meanwhile', 'moreover', 'besides', 'in the beginning',
                  'in the end', 'in addition',
                  'first of all', 'in conclusion', 'then', 'next', 'as soon as', 'later', 'all in all', 'eventually',
                  'once upon a time', 'after', 'before', 'suddenly', 'all of a sudden', 'but then', 'to summarise',
                  'by the end',
                  'by this point'}
    text = ' '.join(filter_punct(get_words_str(fname).lower().split(' ')))
    count = 0
    for seq in sequencers:
        count += text.count(seq)
    return count / len(text.split(' '))


def deictic(fname):
    """
    Returns the proportion of deictic terms in a text passage:
    Numerator --> sum of number of deictic terms
    Denominator -->  # words (excluding punctuations)

    Parameters
    ----------
    count: number deictic terms
    text: text string document without punctuation

    Returns
    -------
    Proportion deictic terms in document (float)
    """
    deictic_terms = {
        ' i ', 'you', 'she', 'he', 'it', 'they', 'we', 'one', 'us', 'him', 'her', 'them',  # personal deixis
        'this', 'that', 'these', 'those', 'here', 'there', 'across the'  # spatial deixis
    }
    text = ' '.join(filter_punct(get_words_str(fname).lower().split(' ')))
    count = 0
    for term in set(deictic_terms):
        count += text.count(term)

    count += time(fname)  # temporal deixis
    return count / len(text.split(' '))


def perceptual_verbs_spatial_prepositions(fname):
    """
    Returns the proportion of sentences including a perceptual verb and spatial preposition in a text passage:

    Parameters
    ----------

    count: number sentences including perceptual verb and spatial preposition
    sentence_count: total number sentences

    Returns
    -------
    Proportion deictic terms in document (float)
    """
    spatial_prepositions = {'across', 'against', 'among', 'around', 'at', 'away', 'before', 'behind', 'behind', 'below',
                            'beneath', 'beside', 'between', 'beyond', 'by', 'in', 'into', 'onto', 'off', 'on',
                            'opposite', 'over', 'under', 'from', 'out', 'to', 'toward', 'inside'}

    df = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.tokens', delimiter='\t', quoting=csv.QUOTE_NONE)
    df_supersense = pd.read_csv(config.BOOK_PATH + fname + '/' + fname + '.supersense', delimiter='\t',
                                quoting=csv.QUOTE_NONE)
    df.fillna("", inplace=True)
    df_supersense.fillna("", inplace=True)

    PV_present, SP_present = False, False

    sentence_count, count = 0, 0
    for row in df.iterrows():
        row = row[1]
        # Check if word is a spatial prepostion
        if row['dependency_relation'] == 'prep' and row['word'] in spatial_prepositions:
            SP_present = True
        if row['POS_tag'] == 'VERB':
            try:
                # Check if verb is a perceptive verb
                if df_supersense.loc[df_supersense['start_token'] == row['token_ID_within_document'],
                                     'supersense_category'].iloc[0] == 'verb.perception':
                    PV_present = True
            except IndexError:
                pass

        # End of sentence
        if row['dependency_relation'] == 'punct' and row['fine_POS_tag'] == '.':
            sentence_count += 1
            if PV_present and SP_present:
                count += 1
            PV_present, SP_present = False, False
    return count / sentence_count
