from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from scipy.stats.stats import pearsonr #spearmanr
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

annotated_df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv')
map_fname_score = dict(annotated_df[['FILENAME', 'avg_overall']].values)



def hyperparameter_tuning(algo_name, X, Y, vectorize_fn, folds, three_class=False):
    """
    Runs the given ML algorithm "algo" with hyperparameter tuning (k-fold CV) - optimises for the best f1.
    Note that the method to vectorize the text is also passed as input in 'vectorize_fn'.

    Parameters
    ----------
    algo_name: name of the ML algorithm ("logreg", "rf")
    X: list of filenames
    Y: corresponding list of labels
    vectorize_fn: function that vectorizes X
    folds: k for stratified k-fold cross validation splits

    Returns
    -------
    best f1-score with the corresponding AUROC, weighted f1, precision, recall, accuracy, AUPRC and hyperparameters
    """
    best_f1 = -1 # can change it to 0.0
    skf = StratifiedKFold(n_splits=folds) # splits the data into stratified folds
    if algo_name == "logreg":
        tuned_parameters = [{'C': [1, 1000], 'penalty': ['l1'], 'solver': ['liblinear'], 'max_iter':[5000]}]
        # tuned_parameters = [{'C': [1, 1000], 'penalty': ['l2'], 'solver': ['lbfgs'], 'max_iter':[5000]}]
        algo = LogisticRegression()

    elif algo_name == "rf":
        tuned_parameters = [{'max_depth': [None, 5, 20]}]
        algo = RandomForestClassifier(n_estimators=500)
        
    elif algo_name == "svm":
        tuned_parameters = [{'C': [0.01, 1], 'kernel': ['linear']},]
        # tuned_parameters = [{'C': [1], 'gamma': ['auto'], 'kernel': ['rbf']}]
        algo = SVC(probability=True)

    param_object = ParameterGrid(tuned_parameters)
    for param_dict in param_object:
        print("Running for parameters:", param_dict)
        algo.set_params(**param_dict) # set the desired hyperparameters
        split_no = 1
        f1s = []; AUROCs = []; weighted_f1s = []; precision_s = []; recall_s = []; accuracies = []; AUPRCs = []
        correlations = []
        for train_indices, test_indices in skf.split(X=np.zeros(len(Y)), y=Y): # only really need Y for splitting
            X_train, X_test = vectorize_fn(train_x=X[train_indices],
                                           test_x=X[test_indices])

            y_train = Y[train_indices]
            y_test = Y[test_indices]


            print("Split number: {} | Train: {} & {} | Test: {} & {}".format(split_no, X_train.shape, y_train.shape, X_test.shape, y_test.shape))
            split_no += 1

            clf = algo.fit(X_train, y_train)
            preds = clf.predict(X_test)
            preds_with_probs = clf.predict_proba(X_test) # for AUROC & AUPRC


            reader_scores = [map_fname_score[fname] for fname in X[test_indices]]

            pear = pearsonr(preds_with_probs[:,1], reader_scores)[0]
            print("Ordering:", clf.classes_, "| Pearson:", pear)
            assert clf.classes_.tolist()[0] == 'NEG' # make sure that the class ordering is ['NEG' 'POS']

            # Compute classification metrics:
            if three_class:
                f1 = f1_score(y_test, preds, average="micro")
                w_f1 = f1_score(y_test, preds, average='weighted')
                precision = precision_score(y_test, preds, average="micro")
                recall = recall_score(y_test, preds, average="micro")
                acc = accuracy_score(y_test, preds)
                auroc, auprc = -1, -1
            else:
                f1 = f1_score(y_test, preds, pos_label="POS")
                w_f1 = f1_score(y_test, preds, average='weighted')
                precision = precision_score(y_test, preds, pos_label="POS")
                recall = recall_score(y_test, preds, pos_label="POS")
                acc = accuracy_score(y_test, preds)
                auroc = roc_auc_score(y_test, preds_with_probs[:,1]) # need to pass probabilities for "greater label"
                auprc = average_precision_score(y_test, preds_with_probs[:,1], pos_label="POS") # need to pass probabilities for positive class

            f1s.append(f1); AUROCs.append(auroc); weighted_f1s.append(w_f1); precision_s.append(precision); recall_s.append(recall); accuracies.append(acc); AUPRCs.append(auprc); correlations.append(pear)



        # Compute mean:
        f1s = np.array(f1s); AUROCs = np.array(AUROCs); weighted_f1s = np.array(weighted_f1s); precision_s = np.array(precision_s); recall_s = np.array(recall_s); accuracies = np.array(accuracies); AUPRCs = np.array(AUPRCs)
        mean_f1 = f1s.mean(); mean_auroc = AUROCs.mean(); mean_weighted_f1 = weighted_f1s.mean(); mean_precision = precision_s.mean(); mean_recall = recall_s.mean(); mean_accuracy = accuracies.mean(); mean_auprc = AUPRCs.mean(); mean_corr = np.array(correlations).mean()

        if mean_f1 > best_f1: # keep track of best f1 and corresponding metrics
            best_f1 = mean_f1
            corresponding_auroc = mean_auroc
            corresponding_weighted_f1 = mean_weighted_f1
            corresponding_precision = mean_precision
            corresponding_recall = mean_recall
            corresponding_accuracy = mean_accuracy
            corresponding_auprc = mean_auprc
            corresponding_params = param_dict
            corresponding_corr = mean_corr

    return round(best_f1, 4), round(corresponding_auroc, 4), round(corresponding_weighted_f1, 4), round(corresponding_precision, 4), round(corresponding_recall, 4), round(corresponding_accuracy, 4), round(corresponding_auprc, 4), round(corresponding_corr, 4), corresponding_params
