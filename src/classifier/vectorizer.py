import features
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os

MAX_FEATURES = None

seed_value = 42  # random seed of 42 for all experiments
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)


def all_feature_categories(train_x, test_x):
    """
    Vectorizes the input text using all feature categories:
    pos1 (max=100) + word1 (max=5000) + dep23 (max=5000) + tense + mood + voice

    We pick the top-performing pos, word, and dep features.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    word_train, word_test = word_unigrams(train_x, test_x, maxfeat=5000)
    dep_train, dep_test = dep_bitri_grams(train_x, test_x, maxfeat=5000)

    tmv_train, tmv_test = tense_mood_voice(train_x, test_x)
    combined_train = np.hstack((tmv_train, dep_train.toarray(), pos_train.toarray(), word_train.toarray()))
    combined_test = np.hstack((tmv_test, dep_test.toarray(), pos_test.toarray(), word_test.toarray()))

    print("Train -- tmv: {} | dep: {} | pos: {} | word: {}".format(tmv_train.shape, dep_train.shape, pos_train.shape,
                                                                   word_train.shape))
    print("Test -- tmv: {} | dep: {} | pos: {} | word: {}".format(tmv_test.shape, dep_test.shape, pos_test.shape,
                                                                  word_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    return combined_train, combined_test


def all_feature_categories_uni(train_x, test_x):
    """
    Vectorizes the input text using all feature categories (simpler model - unigrams only & max=100):
    pos1 (max=100) + word1 (max=100) + dep1 (max=100) + tense + mood + voice

    We pick the top-performing pos, word, and dep features.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    word_train, word_test = word_unigrams(train_x, test_x, maxfeat=100)
    dep_train, dep_test = dep_unigrams(train_x, test_x, maxfeat=100)

    tmvq_train, tmvq_test = tense_mood_voice_quoted(train_x, test_x)
    combined_train = np.hstack((tmvq_train, dep_train.toarray(), pos_train.toarray(), word_train.toarray()))

    if len(test_x) != 0:
        combined_test = np.hstack((tmvq_test, dep_test.toarray(), pos_test.toarray(), word_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- tmv-quoted: {} | dep: {} | pos: {} | word: {}".format(tmvq_train.shape, dep_train.shape,
                                                                          pos_train.shape, word_train.shape))
    print(
        "Test -- tmv-quoted: {} | dep: {} | pos: {} | word: {}".format(tmvq_test.shape, dep_test.shape, pos_test.shape,
                                                                       word_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    return combined_train, combined_test


def pos_unigrams(train_x, test_x, return_feature_names=False, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using part-of-speech unigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(1,1), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(1,1), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)

    if return_feature_names:
        return X_train, X_test, vectorizer.get_feature_names()

    else:
        return X_train, X_test


def pos_bigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the part-of-speech bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(2,2), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(2,2), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def pos_trigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the part-of-speech trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(3,3), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(3,3), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))

    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def pos_bitri_grams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the part-of-speech bigrams and trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    #     vectorizer = CountVectorizer(ngram_range=(2,3), max_features=maxfeat, analyzer='word', encoding='utf-8')
    #     vectorizer = CountVectorizer(ngram_range=(2,3), max_features=maxfeat, token_pattern='\S+', analyzer='word', encoding='utf-8')
    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=maxfeat, token_pattern=r"(?u)\b\w\w+\b|``|\"|\'",
                                 analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_POS_str(x))
    for x in test_x:
        test_sentences.append(features.get_POS_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def dep_unigrams(train_x, test_x, return_feature_names=False, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags unigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    if return_feature_names:
        return X_train, X_test, vectorizer.get_feature_names()
    else:
        return X_train, X_test


def dep_bigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def dep_trigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def dep_bitri_grams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using the dependency-tags bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_dep_str(x))
    for x in test_x:
        test_sentences.append(features.get_dep_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def tense(train_x, test_x):
    """
    Vectorizes the input text using tense features: [temporality, temporal_order]
    
    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append([features.temporality(fname), features.temporal_order(fname)])
    for fname in test_x:
        X_test.append([features.temporality(fname), features.temporal_order(fname)])
    return np.array(X_train), np.array(X_test)


def mood(train_x, test_x):
    """
    Vectorizes the input text using mood features: [setting, concreteness, saying, eventfulness]
    
    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append([features.setting(fname), features.concreteness(fname), features.saying(fname),
                        features.eventfulness(fname)])
    for fname in test_x:
        X_test.append([features.setting(fname), features.concreteness(fname), features.saying(fname),
                       features.eventfulness(fname)])
    return np.array(X_train), np.array(X_test)


def voice(train_x, test_x, coh_kind='seq'):
    """
    Vectorizes the input text using mood features: [agenthood, agency, coherence, feltness]
    
    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append(
            [features.agenthood(fname), features.agency(fname), features.coherence(fname), features.feltness(fname)])
    for fname in test_x:
        X_test.append(
            [features.agenthood(fname), features.agency(fname), features.coherence(fname), features.feltness(fname)])
    return np.array(X_train), np.array(X_test)


def pct_quoted(train_x, test_x):
    """
    Vectorizes the input text using quoted-feature.
    
    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    X_train, X_test = [], []
    for fname in train_x:
        X_train.append([features.compute_quoted_words(fname)])
    for fname in test_x:
        X_test.append([features.compute_quoted_words(fname)])
    return np.array(X_train), np.array(X_test)


def tense_mood_voice(train_x, test_x, return_feature_names=False):
    """
    Vectorizes the input text using all 10 tense/mood/voice features:
    [temporality, temporal_order, setting, concreteness, saying, eventfulness, agenthood, agency, coherence, feltness]

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    return_feature_names: if True, returns the corresponding feature names as well (useful for feature-importance plots)

    Returns
    -------
    X_train, X_test
    """
    tense_train, tense_test = tense(train_x, test_x)
    mood_train, mood_test = mood(train_x, test_x)
    voice_train, voice_test = voice(train_x, test_x)
    combined_train = np.hstack([tense_train, mood_train, voice_train])
    combined_test = np.hstack([tense_test, mood_test, voice_test])

    #     feat_names = ['temporality', 'temporal_order', 'setting', 'concreteness', 'saying', 'eventfulness', 'agenthood', 'agency', 'coherence', 'feltness']
    feat_names = ['temporality', 'setting', 'concreteness', 'saying', 'eventfulness', 'agenthood', 'agency',
                  'coherence', 'feltness']

    if return_feature_names:
        return np.array(combined_train), np.array(combined_test), feat_names
    else:
        return np.array(combined_train), np.array(combined_test)


def tense_mood_voice_quoted(train_x, test_x, return_feature_names=False):
    """
    Vectorizes the input text using all 10 tense/mood/voice features:
    [temporality, temporal_order, setting, concreteness, saying, eventfulness, agenthood, agency, coherence, feltness, quoted]

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    return_feature_names: if True, returns the corresponding feature names as well (useful for feature-importance plots)

    Returns
    -------
    X_train, X_test
    """
    tense_train, tense_test = tense(train_x, test_x)
    mood_train, mood_test = mood(train_x, test_x)
    voice_train, voice_test = voice(train_x, test_x)
    quoted_train, quoted_test = pct_quoted(train_x, test_x)
    combined_train = np.hstack([tense_train, mood_train, voice_train, quoted_train])
    combined_test = np.hstack([tense_test, mood_test, voice_test, quoted_test])

    feat_names = ['temporality', 'temporal_order', 'setting', 'concreteness', 'saying', 'eventfulness', 'agenthood',
                  'agency', 'coherence', 'feltness', 'pct_quoted']

    if return_feature_names:
        return np.array(combined_train), np.array(combined_test), feat_names
    else:
        return np.array(combined_train), np.array(combined_test)


def word_unigrams(train_x, test_x, return_feature_names=False, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW unigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    return_feature_names: if True, returns the corresponding feature names as well (useful for feature-importance plots)

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    if return_feature_names:
        return X_train, X_test, vectorizer.get_feature_names()
    else:
        return X_train, X_test


def word_bigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW bigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def word_trigrams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def word_bitri_grams(train_x, test_x, maxfeat=MAX_FEATURES):
    """
    Vectorizes the input text using BOW bigrams and trigrams. Uses word-level CountVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=maxfeat, analyzer='word', encoding='utf-8')

    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


def pos_tense(train_x, test_x):
    """
    Vectorizes the input text using pos-unigrams with another feature category.
    pos1 (max=100) + tense


    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    other_train, other_test = tense(train_x, test_x)
    combined_train = np.hstack((other_train, pos_train.toarray()))
    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- Other: {} | pos: {}".format(other_train.shape, pos_train.shape))
    print("Test -- Other: {} | pos: {}".format(other_test.shape, pos_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    return combined_train, combined_test


def pos_mood(train_x, test_x):
    """
    Vectorizes the input text using pos-unigrams with another feature category.
    pos1 (max=100) + mood


    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    other_train, other_test = mood(train_x, test_x)
    combined_train = np.hstack((other_train, pos_train.toarray()))
    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- Other: {} | pos: {}".format(other_train.shape, pos_train.shape))
    print("Test -- Other: {} | pos: {}".format(other_test.shape, pos_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    return combined_train, combined_test


def pos_voice(train_x, test_x):
    """
    Vectorizes the input text using pos-unigrams with another feature category.
    pos1 (max=100) + voice


    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    other_train, other_test = voice(train_x, test_x)
    combined_train = np.hstack((other_train, pos_train.toarray()))
    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- Other: {} | pos: {}".format(other_train.shape, pos_train.shape))
    print("Test -- Other: {} | pos: {}".format(other_test.shape, pos_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    return combined_train, combined_test


def pos_tmv(train_x, test_x, return_feature_names=False):
    """
    Vectorizes the input text using pos-unigrams with another feature category.
    pos1 (max=100) + other_funct (tense/mood/voice)


    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    return_feature_names: if True, returns the corresponding feature names as well (useful for feature-importance plots)

    Returns
    -------
    X_train, X_test
    """
    if return_feature_names:
        pos_train, pos_test, pos_feats = pos_unigrams(train_x, test_x, return_feature_names=True, maxfeat=100)
        other_train, other_test, tmv_feats = tense_mood_voice(train_x, test_x, return_feature_names=True)
    else:
        pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
        other_train, other_test = tense_mood_voice(train_x, test_x)

    combined_train = np.hstack((other_train, pos_train.toarray()))

    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- Other: {} | pos: {}".format(other_train.shape, pos_train.shape))
    print("Test -- Other: {} | pos: {}".format(other_test.shape, pos_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    if return_feature_names:
        return combined_train, combined_test, tmv_feats + pos_feats
    return combined_train, combined_test


def pos_tmv_quoted(train_x, test_x, return_feature_names=False):
    """
    Vectorizes the input text using pos-unigrams with another feature category.
    pos1 (max=100) + TMV + quoted


    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames
    return_feature_names: if True, returns the corresponding feature names as well (useful for feature-importance plots)

    Returns
    -------
    X_train, X_test
    """
    if return_feature_names:
        pos_train, pos_test, pos_feats = pos_unigrams(train_x, test_x, return_feature_names=True, maxfeat=100)
        other_train, other_test, tmv_feats = tense_mood_voice_quoted(train_x, test_x, return_feature_names=True)
    else:
        pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
        other_train, other_test = tense_mood_voice_quoted(train_x, test_x)

    combined_train = np.hstack((other_train, pos_train.toarray()))

    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- Other: {} | pos: {}".format(other_train.shape, pos_train.shape))
    print("Test -- Other: {} | pos: {}".format(other_test.shape, pos_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    if return_feature_names:
        return combined_train, combined_test, tmv_feats + pos_feats
    return combined_train, combined_test


def pos_dep_tmv(train_x, test_x, return_feature_names=False):
    """
    Vectorizes the input text using:
    pos1 (max=100) + dep1 (max=100) + tense/mood/voice

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test
    """
    if return_feature_names:
        pos_train, pos_test, pos_feats = pos_unigrams(train_x, test_x, return_feature_names=True, maxfeat=100)
        dep_train, dep_test, dep_feats = dep_unigrams(train_x, test_x, return_feature_names=True, maxfeat=100)
        other_train, other_test, tmv_feats = tense_mood_voice(train_x, test_x, return_feature_names=True)
    else:
        pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
        dep_train, dep_test = dep_unigrams(train_x, test_x, maxfeat=100)
        other_train, other_test = tense_mood_voice(train_x, test_x)

    combined_train = np.hstack((other_train, pos_train.toarray(), dep_train.toarray()))
    if len(test_x) != 0:
        combined_test = np.hstack((other_test, pos_test.toarray(), dep_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    print("Train -- tmv: {} | pos: {} | dep: {}".format(other_train.shape, pos_train.shape, dep_train.shape))
    print("Test -- tmv: {} | pos: {} | dep: {}".format(other_test.shape, pos_test.shape, dep_test.shape))
    print("Combined shape - train: {} & test: {}".format(combined_train.shape, combined_test.shape))
    if return_feature_names:
        return combined_train, combined_test, tmv_feats + pos_feats + dep_feats
    else:
        return combined_train, combined_test


## NEW FEATURES

def doc2vec(train_x, test_x):
    """
    Vectorizes the input text using the Doc2Vec feature

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    doc2vec_train, doc2vec_test = features.doc2vec(train_x, test_x)
    return doc2vec_train, doc2vec_test


def topic_modelling(train_x, test_x):
    """
    Vectorizes the input text using the topic modeling feature

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    tm_train, tm_test = features.topic_modelling(train_x, test_x)
    return tm_train, tm_test


# def top2vec(train_x, test_x):
#     t2v_train, t2v_test = features.top2vec(train_x, test_x)
#     return t2v_train, t2v_test


def improved_concreteness(train_x, test_x):
    """
    Vectorizes the input text using the sequencers feature

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    train_x = np.array([[features.improved_concreteness(fname)] for fname in train_x])
    test_x = np.array([[features.improved_concreteness(fname)] for fname in test_x])
    return train_x, test_x


def ttr(train_x, test_x):
    """
    Vectorizes the input text using the TTR feature

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    train_x = np.array([[features.ttr(fname)] for fname in train_x])
    test_x = np.array([[features.ttr(fname)] for fname in test_x])
    return train_x, test_x


def mtld(train_x, test_x):
    """
    Vectorizes the input text using the MTLD(_BID) feature

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    train_x = np.array([[features.mtld_bid(fname)] for fname in train_x])
    test_x = np.array([[features.mtld_bid(fname)] for fname in test_x])
    return train_x, test_x


def deictic(train_x, test_x):
    """
    Vectorizes the input text using the deictic feature

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    train_x = np.array([[features.deictic(fname)] for fname in train_x])
    test_x = np.array([[features.deictic(fname)] for fname in test_x])
    return train_x, test_x


def sequencers(train_x, test_x):
    """
    Vectorizes the input text using the sequencers feature

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    train_x = np.array([[features.sequencers(fname)] for fname in train_x])
    test_x = np.array([[features.sequencers(fname)] for fname in test_x])
    return train_x, test_x


def perceptual_verbs_spatial_prepositions(train_x, test_x):
    """
    Vectorizes the input text using the perceptual_verbs_spatial_prepositions
    feature

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    train_x = np.array([[features.perceptual_verbs_spatial_prepositions(fname)] for fname in train_x])
    test_x = np.array([[features.perceptual_verbs_spatial_prepositions(fname)] for fname in test_x])
    return train_x, test_x


def adjadv(train_x, test_x):
    """
    Vectorizes the input text using the adjadv feature

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    train_x = np.array([[features.adjadv(fname)] for fname in train_x])
    test_x = np.array([[features.adjadv(fname)] for fname in test_x])
    return train_x, test_x


def tfidf(train_x, test_x):
    """
    Vectorizes the input text using TF-IDF unigrams. Uses word-level TfidfVectorizer.

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    X_train, X_test (sparse matrices)
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), analyzer='word', encoding='utf-8')
    train_sentences, test_sentences = [], []
    for x in train_x:
        train_sentences.append(features.get_words_str(x))
    for x in test_x:
        test_sentences.append(features.get_words_str(x))
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test


## COMBINATIONS ##

def linguistic(train_x, test_x):
    """
    Vectorizes the input text using features sequencers,
    deictic and perceptual_verbs_spatial_prepositions

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    # train_x = np.array([[features.adjadv(fname), features.sequencers(fname), features.deictic(fname), features.perceptual_verbs_spatial_prepositions(fname)] for fname in train_x])
    # test_x = np.array([[features.adjadv(fname), features.sequencers(fname), features.deictic(fname), features.perceptual_verbs_spatial_prepositions(fname)] for fname in test_x])

    train_x = np.array(
        [[features.sequencers(fname), features.deictic(fname), features.perceptual_verbs_spatial_prepositions(fname)]
         for fname in train_x])

    test_x = np.array(
        [[features.sequencers(fname), features.deictic(fname), features.perceptual_verbs_spatial_prepositions(fname)]
         for fname in test_x])
    return train_x, test_x


def doc2vec_tm(train_x, test_x):
    """
    Vectorizes the input text using Doc2Vec
    and topic modeling

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    d2v_train, d2v_test = doc2vec(train_x, test_x)
    tm_train, tm_test = topic_modelling(train_x, test_x)
    combined_train = np.hstack((d2v_train, tm_train))
    combined_test = np.hstack((d2v_test, tm_test))
    return combined_train, combined_test


def doc2vec_tm_ttr(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_tm
    and TTR

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    ttr_train, ttr_test = ttr(train_x, test_x)
    other_train, other_test = doc2vec_tm(train_x, test_x)
    combined_train = np.hstack((ttr_train, other_train))
    combined_test = np.hstack((ttr_test, other_test))
    return combined_train, combined_test


def doc2vec_tm_mtld(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_tm
    and MTLD

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    mtld_train, mtld_test = mtld(train_x, test_x)
    other_train, other_test = doc2vec_tm(train_x, test_x)
    combined_train = np.hstack((mtld_train, other_train))
    combined_test = np.hstack((mtld_test, other_test))
    return combined_train, combined_test


def doc2vec_tm_ttr_mtld(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_tm_ttr
    and MTLD

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    mtld_train, mtld_test = mtld(train_x, test_x)
    other_train, other_test = doc2vec_tm_ttr(train_x, test_x)
    combined_train = np.hstack((mtld_train, other_train))
    combined_test = np.hstack((mtld_test, other_test))
    return combined_train, combined_test


def doc2vec_tm_ttr_concreteness(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_tm_ttr
    and improved concreteness

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    conc_train, conc_test = improved_concreteness(train_x, test_x)
    other_train, other_test = doc2vec_tm_ttr(train_x, test_x)
    combined_train = np.hstack((conc_train, other_train))
    combined_test = np.hstack((conc_test, other_test))
    return combined_train, combined_test


def doc2vec_ttr(train_x, test_x):
    """
    Vectorizes the input text using Doc2Vec and TTR

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    d2v_train, d2v_test = doc2vec(train_x, test_x)
    ttr_train, ttr_test = ttr(train_x, test_x)
    combined_train = np.hstack((ttr_train, d2v_train))
    combined_test = np.hstack((ttr_test, d2v_test))
    return combined_train, combined_test


def doc2vec_mtld(train_x, test_x):
    """
    Vectorizes the input text using Doc2Vec and MTLD

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    d2v_train, d2v_test = doc2vec(train_x, test_x)
    mtld_train, mtld_test = mtld(train_x, test_x)
    combined_train = np.hstack((mtld_train, d2v_train))
    combined_test = np.hstack((mtld_test, d2v_test))
    return combined_train, combined_test


def doc2vec_ttr_mtld(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_ttr
    and MTLD

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    mtld_train, mtld_test = mtld(train_x, test_x)
    other_train, other_test = doc2vec_ttr(train_x, test_x)
    combined_train = np.hstack((mtld_train, other_train))
    combined_test = np.hstack((mtld_test, other_test))
    return combined_train, combined_test


def doc2vec_concreteness(train_x, test_x):
    """
    Vectorizes the input text using Doc2Vec
    and improved concreteness

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    d2v_train, d2v_test = doc2vec(train_x, test_x)
    conc_train, conc_test = improved_concreteness(train_x, test_x)
    combined_train = np.hstack((conc_train, d2v_train))
    combined_test = np.hstack((conc_test, d2v_test))
    return combined_train, combined_test


def doc2vec_ttr_concreteness(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_ttr
    and improved concreteness

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    conc_train, conc_test = improved_concreteness(train_x, test_x)
    other_train, other_test = doc2vec_ttr(train_x, test_x)
    combined_train = np.hstack((conc_train, other_train))
    combined_test = np.hstack((conc_test, other_test))
    return combined_train, combined_test


def doc2vec_ttr_linguistic(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_ttr
    and linguistic features

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    ling_train, ling_test = linguistic(train_x, test_x)
    other_train, other_test = doc2vec_ttr(train_x, test_x)
    combined_train = np.hstack((ling_train, other_train))
    combined_test = np.hstack((ling_test, other_test))
    return combined_train, combined_test


def doc2vec_ttr_concreteness_linguistic(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_ttr_concreteness
    and linguistic features

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    ling_train, ling_test = linguistic(train_x, test_x)
    other_train, other_test = doc2vec_ttr_concreteness(train_x, test_x)
    combined_train = np.hstack((ling_train, other_train))
    combined_test = np.hstack((ling_test, other_test))
    return combined_train, combined_test


def doc2vec_ttr_concreteness_linguistic_tfidf(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_ttr_concreteness_linguistic
    and tfidf

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    tfidf_train, tfidf_test = tfidf(train_x, test_x)
    other_train, other_test = doc2vec_ttr_concreteness_linguistic(train_x, test_x)
    combined_train = np.hstack((tfidf_train.toarray(), other_train))
    combined_test = np.hstack((tfidf_test.toarray(), other_test))
    return combined_train, combined_test


def tfidf_concreteness(train_x, test_x):
    """
    Vectorizes the input text using TfidfVectorizer
    and improved concreteness

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    tfidf_train, tfidf_test = tfidf(train_x, test_x)
    conc_train, conc_test = improved_concreteness(train_x, test_x)
    combined_train = np.hstack((tfidf_train.toarray(), conc_train))
    combined_test = np.hstack((tfidf_test.toarray(), conc_test))
    return combined_train, combined_test


def tm_ttr_concreteness_linguistic(train_x, test_x):
    """
    Vectorizes the input text using linguistic features, ttr,
    improved_concreteness and topic_modeling

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    ling_train, ling_test = linguistic(train_x, test_x)
    ttr_train, ttr_test = ttr(train_x, test_x)
    conc_train, conc_test = improved_concreteness(train_x, test_x)
    tm_train, tm_test = topic_modelling(train_x, test_x)

    combined_train = np.hstack((ling_train, ttr_train, conc_train, tm_train))
    combined_test = np.hstack((ling_test, ttr_test, conc_test, tm_test))
    return combined_train, combined_test


def doc2vec_concreteness_nlp(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_concreteness
    and dep/pos/word uni-/bi-/tri-/bitrigrams

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    # word_train, word_test = dep_unigrams(train_x, test_x)  # bigrams, trigrams, bitrigrams
    # word_train, word_test = pos_unigrams(train_x, test_x) # bigrams, trigrams, bitrigrams
    word_train, word_test = word_unigrams(train_x, test_x)  # bigrams, trigrams, bitrigrams
    other_train, other_test = doc2vec_concreteness(train_x, test_x)
    combined_train = np.hstack((word_train.toarray(), other_train))
    combined_test = np.hstack((word_test.toarray(), other_test))
    return combined_train, combined_test


def doc2vec_ttr_concreteness_nlp(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_ttr_concreteness
    and dep/pos/word uni-/bi-/tri-/bitrigrams

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    # word_train, word_test = dep_unigrams(train_x, test_x)  # bigrams, trigrams, bitrigrams
    # word_train, word_test = pos_unigrams(train_x, test_x) # bigrams, trigrams, bitrigrams
    word_train, word_test = word_unigrams(train_x, test_x)  # bigrams, trigrams, bitrigrams
    other_train, other_test = doc2vec_ttr_concreteness(train_x, test_x)
    combined_train = np.hstack((word_train.toarray(), other_train))
    combined_test = np.hstack((word_test.toarray(), other_test))
    return combined_train, combined_test


def doc2vec_ttr_concreteness_linguistic_nlp(train_x, test_x):
    """
    Vectorizes the input text using function doc2vec_ttr_concreteness_linguistic
    and dep/pos/word uni-/bi-/tri-/bitrigrams

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    # word_train, word_test = dep_unigrams(train_x, test_x)  # bigrams, trigrams, bitrigrams
    # word_train, word_test = pos_unigrams(train_x, test_x) # bigrams, trigrams, bitrigrams
    word_train, word_test = word_unigrams(train_x, test_x)  # bigrams, trigrams, bitrigrams
    other_train, other_test = doc2vec_ttr_concreteness_linguistic(train_x, test_x)
    combined_train = np.hstack((word_train.toarray(), other_train))
    combined_test = np.hstack((word_test.toarray(), other_test))
    return combined_train, combined_test


def ttr_concreteness_linguistic(train_x, test_x):
    """
    Vectorizes the input text using linguistic features, ttr
    and improved concreteness

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    ling_train, ling_test = linguistic(train_x, test_x)
    ttr_train, ttr_test = ttr(train_x, test_x)
    conc_train, conc_test = improved_concreteness(train_x, test_x)
    combined_train = np.hstack((ling_train, ttr_train, conc_train))
    combined_test = np.hstack((ling_test, ttr_test, conc_test))
    return combined_train, combined_test


def pos_ttr_concreteness_linguistic(train_x, test_x):
    """
    Vectorizes the input text using function ttr_concreteness_linguistic
    and pos_unigrams

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    other_train, other_test = ttr_concreteness_linguistic(train_x, test_x)
    combined_train = np.hstack((pos_train.toarray(), other_train))
    combined_test = np.hstack((pos_test.toarray(), other_test))
    return combined_train, combined_test


def tfidf_ttr_concreteness_linguistic(train_x, test_x):
    """
    Vectorizes the input text using function ttr_concreteness_linguistic
    and TfidfVectorizer

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    tfidf_train, tfidf_test = tfidf(train_x, test_x)
    other_train, other_test = ttr_concreteness_linguistic(train_x, test_x)
    combined_train = np.hstack((tfidf_train.toarray(), other_train))
    combined_test = np.hstack((tfidf_test.toarray(), other_test))
    return combined_train, combined_test


def pos_dep_doc2vec(train_x, test_x):
    """
    Vectorizes the input text using pos_unigrams, dep_unigrams
    and doc2vec

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    dep_train, dep_test = dep_unigrams(train_x, test_x, maxfeat=100)
    d2v_train, d2v_test = doc2vec(train_x, test_x)

    combined_train = np.hstack((d2v_train, pos_train.toarray(), dep_train.toarray()))
    if len(test_x) != 0:
        combined_test = np.hstack((d2v_test, pos_test.toarray(), dep_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    return combined_train, combined_test


def pos_dep_tm(train_x, test_x):
    """
    Vectorizes the input text using pos_unigrams, dep_unigrams
    and topic modeling

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    pos_train, pos_test = pos_unigrams(train_x, test_x, maxfeat=100)
    dep_train, dep_test = dep_unigrams(train_x, test_x, maxfeat=100)
    tm_train, tm_test = topic_modelling(train_x, test_x)

    combined_train = np.hstack((tm_train, pos_train.toarray(), dep_train.toarray()))
    if len(test_x) != 0:
        combined_test = np.hstack((tm_test, pos_test.toarray(), dep_test.toarray()))
    else:  # test set is empty
        combined_test = np.array([])

    return combined_train, combined_test


def pos_dep_doc2vec_tm(train_x, test_x):
    """
    Vectorizes the input text using function pos_dep_doc2vec
    and topic modeling

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    train_vec, test_vec = pos_dep_doc2vec(train_x, test_x)
    tm_train, tm_test = topic_modelling(train_x, test_x)
    combined_train = np.hstack((train_vec, tm_train))
    if len(test_x) != 0:
        combined_test = np.hstack((test_vec, tm_test))
    else:  # test set is empty
        combined_test = np.array([])
    return combined_train, combined_test


def pos_dep_doc2vec_concreteness(train_x, test_x):
    """
    Vectorizes the input text using function pos_dep_doc2vec
    and improved concreteness

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    train_vec, test_vec = pos_dep_doc2vec(train_x, test_x)
    conc_train, conc_test = improved_concreteness(train_x, test_x)
    combined_train = np.hstack((train_vec, conc_train))
    if len(test_x) != 0:
        combined_test = np.hstack((test_vec, conc_test))
    else:  # test set is empty
        combined_test = np.array([])
    return combined_train, combined_test


def pos_dep_doc2vec_concreteness_ld(train_x, test_x):
    """
    Vectorizes the input text using function pos_dep_doc2vec_concreteness
    and TTR

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    train_vec, test_vec = pos_dep_doc2vec_concreteness(train_x, test_x)
    ld_train, ld_test = ttr(train_x, test_x)
    combined_train = np.hstack((train_vec, ld_train))
    if len(test_x) != 0:
        combined_test = np.hstack((test_vec, ld_test))
    else:  # test set is empty
        combined_test = np.array([])
    return combined_train, combined_test


def pos_dep_doc2vec_concreteness_tm_ld(train_x, test_x):
    """
    Vectorizes the input text using function pos_dep_doc2vec_concreteness
    and topic_modeling and TTR

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    train_vec, test_vec = pos_dep_doc2vec_concreteness(train_x, test_x)
    tm_train, tm_test = topic_modelling(train_x, test_x)
    ld_train, ld_test = ttr(train_x, test_x)
    combined_train = np.hstack((train_vec, tm_train, ld_train))
    if len(test_x) != 0:
        combined_test = np.hstack((test_vec, tm_test, ld_test))
    else:  # test set is empty
        combined_test = np.array([])
    return combined_train, combined_test


def pos_dep_doc2vec_concreteness_tm_ld_linguistic(train_x, test_x):
    """
    Vectorizes the input text using function pos_dep_doc2vec_concreteness_tm_ld
    and linguistic

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    train_vec, test_vec = pos_dep_doc2vec_concreteness_tm_ld(train_x, test_x)
    ling_train, ling_test = linguistic(train_x, test_x)
    combined_train = np.hstack((train_vec, ling_train))
    if len(test_x) != 0:
        combined_test = np.hstack((test_vec, ling_test))
    else:  # test set is empty
        combined_test = np.array([])
    return combined_train, combined_test


def pos_dep_doc2vec_concreteness_ld_linguistic(train_x, test_x):
    """
    Vectorizes the input text using function pos_dep_doc2vec_concreteness_ld
    and linguistic

    Parameters
    ----------
    train_x: list of train filenames
    test_x: list of test filenames

    Returns
    -------
    combined_train, combined_test
    """
    train_vec, test_vec = pos_dep_doc2vec_concreteness_ld(train_x, test_x)
    ling_train, ling_test = linguistic(train_x, test_x)
    combined_train = np.hstack((train_vec, ling_train))
    if len(test_x) != 0:
        combined_test = np.hstack((test_vec, ling_test))
    else:  # test set is empty
        combined_test = np.array([])
    return combined_train, combined_test