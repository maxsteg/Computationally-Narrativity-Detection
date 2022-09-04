import pandas as pd
import numpy as np


def load_annotated_data(threshold):
    """
    Loads Annotated Dataset (2 class). Returns X & Y.
    """

    # Data from Piper 2022
    # path = '../../dataset/MinNarrative_ReaderData_Final.csv'
    path = '../../dataset/Universal_Annotation_Results_Selection.csv'
    df = pd.read_csv(path)
    print("Loading annotated data from:", path)
    X, Y = [], []
    for fname, score in df[['FILENAME', 'avg_overall']].values:
        if score > threshold:
            Y.append('POS')
        else:
            Y.append('NEG')
        X.append(fname)
    return np.array(X), np.array(Y)


def load_annotated_data_3class():
    """
    Loads Annotated Dataset (3 class). Returns X & Y.
    (1-2) = NEG
    [2-3] = NEUTRAL
    (3,5] = POS
    """
    df = pd.read_csv('../../dataset/Universal_Annotation_Results_Selection.csv', delimiter=',')
    # df = pd.read_csv('../../dataset/MinNarrative_ReaderData_Final.csv', delimiter=',')

    # 3 class where: are the three classes.
    X, Y = [], []
    for fname, score in df[['FILENAME', 'avg_overall']].values:
        if score < 2:
            Y.append('NEG')
        elif score >= 2 and score <= 3:
            Y.append('NEUTRAL')
        elif score > 3:
            Y.append('POS')
        X.append(fname)
    return np.array(X), np.array(Y)
