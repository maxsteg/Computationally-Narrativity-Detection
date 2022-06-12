import pandas as pd

BOOK_PATH = "/home/max/PycharmProjects/Thesis/dataset/BookNLP_output/"

PRONOUNS = ['i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'his', 'hers', 'my', 'mine', 'our', 'ours', 'your', 'yours', 'their', 'theirs',
            'thy', 'thee', 'thou']

HELPING_VERBS = ['am', 'is', 'are', 'was', 'were', 'being', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
                 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could']

# Concreteness:
MAP_WORD_CONC = dict(pd.read_csv('../../dataset/Concreteness_ratings_Brysbaert_et_al_BRM.txt', delimiter='\t')[
                         ['Word', 'Conc.M']].values)


def improved_concreteness():
    df = pd.read_csv('../../dataset/MultiwordExpression_Concreteness_Ratings.csv', delimiter=',')
    EXTENDED_MAP_WORD_CONC = dict(
        pd.read_csv('../../dataset/Concreteness_ratings_Brysbaert_et_al_BRM.txt', delimiter='\t')[
            ['Word', 'Conc.M']].values)
    MULTI_WORD = dict(df[df['Mean_C'].notna()][['Expression', 'Mean_C']].values)
    EXTENDED_MAP_WORD_CONC.update(MULTI_WORD)
    return EXTENDED_MAP_WORD_CONC


EXTENDED_MAP_WORD_CONC = improved_concreteness()
