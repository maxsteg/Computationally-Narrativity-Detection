from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, \
    average_precision_score
from scipy.stats.stats import pearsonr  # spearmanr
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn import linear_model, metrics

annotated_df = pd.read_csv('../../dataset/Universal_Annotation_Results_Selection.csv')
# annotated_df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv')
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
    best_f1 = -1  # can change it to 0.0
    best_r2 = -100000
    skf = StratifiedKFold(n_splits=folds)  # splits the data into stratified folds
    if algo_name == "logreg":
        tuned_parameters = [{'C': [1, 1000], 'penalty': ['l1'], 'solver': ['liblinear'], 'max_iter': [5000]}]
        algo = LogisticRegression()

    elif algo_name == "rf":
        tuned_parameters = [{'max_depth': [None, 5, 20]}]
        algo = RandomForestClassifier(n_estimators=500)

    elif algo_name == "svm":
        tuned_parameters = [{'C': [0.01, 1], 'kernel': ['linear']}, ]
        # tuned_parameters = [{'C': [1], 'gamma': ['auto'], 'kernel': ['rbf']}]
        algo = SVC(probability=True)
    elif algo_name == 'linearregression':
        tuned_parameters = [{}]
        algo = linear_model.LinearRegression()
    elif algo_name == 'elasticnet':
        tuned_parameters = [{'alpha': [0.1, 0.05, 0.001]}]
        algo = linear_model.ElasticNet()
    elif algo_name == 'huberregressor':
        tuned_parameters = [{}]
        algo = linear_model.HuberRegressor()
    elif algo_name == 'ridge':
        tuned_parameters = [{}]
        algo = linear_model.Ridge()
    elif algo_name == 'lasso':
        tuned_parameters = [{'alpha':[0.1, 0.05, 0.001]}]
        algo = linear_model.Lasso()
    elif algo_name == 'theilsenregressor':
        tuned_parameters = [{}]
        algo = linear_model.TheilSenRegressor(random_state=0, n_jobs=-1)

    param_object = ParameterGrid(tuned_parameters)
    for param_dict in param_object:
        print("Running for parameters:", param_dict)
        algo.set_params(**param_dict)  # set the desired hyperparameters
        split_no = 1
        f1s = [];
        AUROCs = [];
        weighted_f1s = [];
        precision_s = [];
        recall_s = [];
        accuracies = [];
        AUPRCs = []
        correlations = []
        r2s =  [];
        for train_indices, test_indices in skf.split(X=np.zeros(len(Y)), y=Y):  # only really need Y for splitting
            X_train, X_test = vectorize_fn(train_x=X[train_indices],
                                           test_x=X[test_indices])

            y_train = Y[train_indices]
            y_test = Y[test_indices]

            print("Split number: {} | Train: {} & {} | Test: {} & {}".format(split_no, X_train.shape, y_train.shape,
                                                                             X_test.shape, y_test.shape))

            # Rescale Reader Annotation Scores from 1 to 5, to 0 to 1.
            y_train = [((1 - 0) * (map_fname_score[fname] - 1) / (5 - 1)) for fname in X[train_indices]]


            try:
                if type(X_train).__name__ == 'csr_matrix':
                    X_train = X_train.toarray()
            except:
                pass

            clf = algo.fit(X_train, y_train)
            preds_continuous = np.array([float(x) for x in clf.predict(X_test)])

            # Limit out-of-range predictions (< 0, > 1)
            preds_continuous[preds_continuous < 0.0] = 0
            preds_continuous[preds_continuous > 1.0] = 1

            # Rescale Reader Annotation Scores from 1 to 5, to 0 to 1.
            reader_scores = [((1 - 0) * (map_fname_score[fname] - 1) / (5 - 1)) for fname in X[test_indices]]

            pear = pearsonr(preds_continuous, reader_scores)[0]

            print("Pearson:", pear)
            preds = np.where(preds_continuous > 0.5, 'POS', 'NEG')

            print(preds_continuous)
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
                auroc = roc_auc_score(y_test, preds_continuous)
                auprc = average_precision_score(y_test, preds_continuous, pos_label="POS")
                r2 = metrics.r2_score(reader_scores, preds_continuous)

            f1s.append(f1);
            AUROCs.append(auroc);
            weighted_f1s.append(w_f1);
            precision_s.append(precision);
            recall_s.append(recall);
            accuracies.append(acc);
            AUPRCs.append(auprc);
            correlations.append(pear)
            r2s.append(r2);

        r2s = np.array(r2s)
        mean_r2 = r2s.mean()
        # Compute mean:
        f1s = np.array(f1s);
        AUROCs = np.array(AUROCs);
        weighted_f1s = np.array(weighted_f1s);
        precision_s = np.array(precision_s);
        recall_s = np.array(recall_s);
        accuracies = np.array(accuracies);
        AUPRCs = np.array(AUPRCs)
        mean_f1 = f1s.mean();
        mean_auroc = AUROCs.mean();
        mean_weighted_f1 = weighted_f1s.mean();
        mean_precision = precision_s.mean();
        mean_recall = recall_s.mean();
        mean_accuracy = accuracies.mean();
        mean_auprc = AUPRCs.mean();
        mean_corr = np.array(correlations).mean()

        if mean_r2 > best_r2:  # keep track of best f1 and corresponding metrics
            best_r2 = mean_r2
            best_f1 = mean_f1
            corresponding_auroc = mean_auroc
            corresponding_weighted_f1 = mean_weighted_f1
            corresponding_precision = mean_precision
            corresponding_recall = mean_recall
            corresponding_accuracy = mean_accuracy
            corresponding_auprc = mean_auprc
            corresponding_params = param_dict
            corresponding_corr = mean_corr

    return round(best_r2, 4), round(best_f1, 4), round(corresponding_auroc, 4), round(corresponding_weighted_f1, 4), round(
        corresponding_precision, 4), round(corresponding_recall, 4), round(corresponding_accuracy, 4), round(
        corresponding_auprc, 4), round(corresponding_corr, 4), corresponding_params
