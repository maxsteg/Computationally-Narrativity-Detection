"""
- Run multiple experiments for different feature-categories.
- K-fold cross validated hyperparamter tuning for Logistic Regression
- Report the best f1-score along with precision, recall, AUROC, AUPRC, Weighted f1, and accuracy.

- Runs on main training dataset (~13k) with/without misclassifications.
- Runs on 2-class Reader-Annotated dataset. See main_3class for 3-class Reader-Annotated dataset.
"""

import os
import random
import numpy as np
import vectorizer
import data_loader
import tuning
from collections import Counter

seed_value = 42  # random seed of 42 for all experiments
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


def run_experiments(algo, X, Y, name):
    """
    Run experiments for different feature categories as specified by "name".
    """

    # Baselines
    if name == 'pos1':
        funct = vectorizer.pos_unigrams
    elif name == 'pos2':
        funct = vectorizer.pos_bigrams
    elif name == 'pos3':
        funct = vectorizer.pos_trigrams
    elif name == 'pos23':
        funct = vectorizer.pos_bitri_grams

    elif name == 'word1':
        funct = vectorizer.word_unigrams
    elif name == 'word2':
        funct = vectorizer.word_bigrams
    elif name == 'word3':
        funct = vectorizer.word_trigrams
    elif name == 'word23':
        funct = vectorizer.word_bitri_grams

    elif name == 'dep1':
        funct = vectorizer.dep_unigrams
    elif name == 'dep2':
        funct = vectorizer.dep_bigrams
    elif name == 'dep3':
        funct = vectorizer.dep_trigrams
    elif name == 'dep23':
        funct = vectorizer.dep_bitri_grams

    elif name == 'tense':
        funct = vectorizer.tense
    elif name == 'mood':
        funct = vectorizer.mood
    elif name == 'voice':
        funct = vectorizer.voice
    elif name == 'tense_mood_voice':
        funct = vectorizer.tense_mood_voice

    elif name == 'pos_tense':
        funct = vectorizer.pos_tense
    elif name == 'pos_mood':
        funct = vectorizer.pos_mood
    elif name == 'pos_voice':
        funct = vectorizer.pos_voice
    elif name == 'pos_tense_mood_voice':
        funct = vectorizer.pos_tmv
    elif name == 'pos_tense_mood_voice_quoted':  # pos1 (max=100)
        funct = vectorizer.pos_tmv_quoted


    elif name == 'pos_dep_tense_mood_voice':  # pos1 (max=100) + dep1 (max=100) + tense + mood + voice
        funct = vectorizer.pos_dep_tmv
    elif name == 'all_categories':  # pos1 (max=100) + word1 (max=100) + dep1 (max=100) + tense + mood + voice + pct_quoted
        funct = vectorizer.all_feature_categories_uni
    elif name == 'all_categories_best':  # pos1 (max=100) + word1 (max=5000) + dep23 (max=5000) + tense + mood + voice
        funct = vectorizer.all_feature_categories

    # Individual feature (categories)

    elif name == 'improved_concreteness':
        funct = vectorizer.improved_concreteness
    elif name == 'perceptual_verbs_spatial_prepositions':
        funct = vectorizer.perceptual_verbs_spatial_prepositions
    elif name == 'mtld':
        funct = vectorizer.mtld
    elif name == 'ttr':
        funct = vectorizer.ttr
    elif name == 'deictic':
        funct = vectorizer.deictic
    elif name == 'adjadv':
        funct = vectorizer.adjadv
    elif name == 'sequencers':
        funct = vectorizer.sequencers
    elif name == 'doc2vec':
        funct = vectorizer.doc2vec
    elif name == 'tm':
        funct = vectorizer.topic_modelling
    elif name == 'tfidf':
        funct = vectorizer.tfidf
    elif name == 'linguistic':
        funct = vectorizer.linguistic


    # Feature combinations
    elif name == 'ttr_concreteness_linguistic':
        funct = vectorizer.ttr_concreteness_linguistic
    elif name == 'tfidf_ttr_concreteness_linguistic':
        funct = vectorizer.tfidf_ttr_concreteness_linguistic
    elif name == 'doc2vec_tm':
        funct = vectorizer.doc2vec_tm
    elif name == 'doc2vec_tm_ttr':
        funct = vectorizer.doc2vec_tm_ttr
    elif name == 'doc2vec_tm_ttr_concreteness':
        funct = vectorizer.doc2vec_tm_ttr_concreteness
    elif name == 'doc2vec_ttr':
        funct = vectorizer.doc2vec_ttr
    elif name == 'doc2vec_ttr_mtld':
        funct = vectorizer.doc2vec_ttr_mtld
    elif name == 'doc2vec_concreteness':
        funct = vectorizer.doc2vec_concreteness
    elif name == 'doc2vec_ttr_concreteness':
        funct = vectorizer.doc2vec_ttr_concreteness
    elif name == 'doc2vec_ttr_linguistic':
        funct = vectorizer.doc2vec_ttr_linguistic
    elif name == 'doc2vec_ttr_concreteness_linguistic':
        funct = vectorizer.doc2vec_ttr_concreteness_linguistic
    elif name == 'tm_ttr_concreteness_linguistic':
        funct = vectorizer.tm_ttr_concreteness_linguistic
    elif name == 'doc2vec_ttr_concreteness_linguistic_tfidf':
        funct = vectorizer.doc2vec_ttr_concreteness_linguistic_tfidf
    elif name == 'tfidf_concreteness':
        funct = vectorizer.tfidf_concreteness


    # Other experiments
    elif name == 'doc2vec_concreteness_nlp':
        funct = vectorizer.doc2vec_concreteness_nlp
    elif name == 'doc2vec_ttr_concreteness_nlp':
        funct = vectorizer.doc2vec_ttr_concreteness_nlp
    elif name == 'doc2vec_ttr_concreteness_linguistic_nlp':
        funct = vectorizer.doc2vec_ttr_concreteness_linguistic_nlp
    elif name == 'pos_ttr_concreteness_linguistic':  # pos1 (max=100)
        funct = vectorizer.pos_ttr_concreteness_linguistic
    elif name == 'pos_dep_doc2vec':
        funct = vectorizer.pos_dep_doc2vec
    elif name == 'pos_dep_tm':
        funct = vectorizer.pos_dep_tm
    elif name == 'pos_dep_doc2vec_tm':
        funct = vectorizer.pos_dep_doc2vec_tm
    elif name == 'pos_dep_doc2vec_concreteness':
        funct = vectorizer.pos_dep_doc2vec_concreteness
    elif name == 'pos_dep_doc2vec_concreteness_ld':
        funct = vectorizer.pos_dep_doc2vec_concreteness_ld
    elif name == 'pos_dep_doc2vec_concreteness_tm_ld':
        funct = vectorizer.pos_dep_doc2vec_concreteness_tm_ld
    elif name == 'pos_dep_doc2vec_concreteness_ld_linguistic':
        funct = vectorizer.pos_dep_doc2vec_concreteness_ld_linguistic
    elif name == 'pos_dep_doc2vec_concreteness_tm_ld_linguistic':
        funct = vectorizer.pos_dep_doc2vec_concreteness_tm_ld
    elif name == 'doc2vec_tm_mtld':
        funct = vectorizer.doc2vec_tm_mtld
    elif name == 'doc2vec_tm_ttr_mtld':
        funct = vectorizer.doc2vec_tm_ttr_mtld
    elif name == 'doc2vec_mtld':
        funct = vectorizer.doc2vec_mtld

    # Extra
    elif name == 'tfidf_doc2vec':
        funct = vectorizer.tfidf_doc2vec
    elif name == 'tfidf_doc2vec_concreteness':
        funct = vectorizer.tfidf_doc2vec_concreteness
    elif name == 'tfidf_doc2vec_word1':
        funct = vectorizer.tfidf_doc2vec_word1
    elif name == 'tfidf_word1_concreteness':
        funct = vectorizer.tfidf_doc2vec_word1
    elif name == 'word1_doc2vec_concreteness':
        funct = vectorizer.tfidf_doc2vec_word1
    elif name == 'word1_doc2vec_tfidf_concreteness':
        funct = vectorizer.tfidf_doc2vec_word1


    r2, f1, auc, weighted_f1, prec, rec, accuracy, auprc, corr, params = tuning.hyperparameter_tuning(algo, X, Y, funct,
                                                                                                  NUMBER_OF_FOLDS,
                                                                                                  three_class)

    print("R2:", r2)
    results_file.write(
        str(r2) + '\t' + str(f1) + '\t' + str(auc) + '\t' + str(weighted_f1) + '\t' + str(prec) + '\t' + str(rec) + '\t' + str(
            accuracy) + '\t' + str(auprc) + '\t' + str(corr) + '\t' + str(params) + '\n')


def main(three_class=False):
    """
    Run experiments for different feature-categories and different data-subsets.
    """
    kind = '5S'
    for feature_name in features:
        print("\n\n-----------------------\nRUNNING FOR: Kind =", kind, "| Feature =", feature_name)
        # X, Y = data_loader.load_data(discard_genres=['OPINION'], remove_annotated_passages=True, remove_mispreds=True)
        X, Y = data_loader.load_annotated_data(threshold=2.5)
        print(len(X), len(Y))
        if three_class:
            X, Y = data_loader.load_annotated_data_3class()
        print("\nX: {} | Y: {} | Distribution: {} | Y preview: {}".format(len(X), len(Y), Counter(Y), Y[:3]))
        results_file.write(kind + '_' + str(len(X)) + '\t' + feature_name + '\t')
        run_experiments(algo_name, X, Y, feature_name)


if __name__ == '__main__':
    algo_name = 'theilsenregressor'  # 'rf' or 'logreg' or 'svm' or '
    NUMBER_OF_FOLDS = 5

    three_class = False

    # print("Running {}-fold CV | Algo = {} | Max-features = {}".format(NUMBER_OF_FOLDS, algo_name, vectorizer.MAX_FEATURES))
    print("Running {}-fold CV | Algo = {} | Max-features = {}".format(NUMBER_OF_FOLDS, algo_name,
                                                                      vectorizer.MAX_FEATURES))
    results_path = './results/' + algo_name + '__' + str(NUMBER_OF_FOLDS) + '_foldcv.txt'  # name of output file
    print("\n-------\nResults path:", results_path, "\n\n")
    results_file = open(results_path, "w")
    results_file.write(
        "Data\tFeature\tR2-score\tF1-score\tAUROC\tWeighted F1\tPrecision\tRecall\tAccuracy\tAUPRC\tCorrelation\tParameters\n")

    # For all possibilities, see function 'run_experiments'

    # Baselines
    # features = ['pos1', 'pos2', 'pos3', 'pos23', 'word1', 'word2', 'word3', 'word23', 'dep1', 'dep2', 'dep3', 'dep23', 'tense', 'mood', 'voice', 'tense_mood_voice', 'pos_tense', 'pos_mood', 'pos_voice', 'pos_tense_mood_voice', 'pos_tense_mood_voice_quoted', 'pos_dep_tense_mood_voice', 'all_categories', 'all_categories_best']

    # New individual features and input representations
    # features = ['linguistic', 'improved_concreteness', 'ttr', 'deictic', 'sequencers', 'perceptual_verbs_spatial_prepositions', 'doc2vec', 'tm', 'tfidf']

    # Feature combinations
    # features = ['ttr_concreteness_linguistic', 'tfidf_ttr_concreteness_linguistic', 'doc2vec_tm', 'doc2vec_tm_ttr',
    #             'doc2vec_tm_ttr_concreteness', 'doc2vec_ttr', 'doc2vec_ttr_mtld', 'doc2vec_concreteness',
    #             'doc2vec_ttr_concreteness', 'doc2vec_ttr_linguistic', 'doc2vec_ttr_concreteness_linguistic',
    #             'tfidf_concreteness', 'doc2vec_ttr_concreteness_linguistic_tfidf', 'tm_ttr_concreteness_linguistic']

    features = ['word1', 'tfidf', 'doc2vec', 'improved_concreteness', 'doc2vec_concreteness', 'tfidf_concreteness',
                'doc2vec_ttr_concreteness','tense_mood_voice', 'tfidf_doc2vec', 'tfidf_doc2vec_concreteness', 'tfidf_doc2vec_word1', 'tfidf_word1_concreteness',
                 'word1_doc2vec_concreteness', 'word1_doc2vec_tfidf_concreteness']

    # features = ['tfidf']

    main()
