# This code creates a LDA model, using the complete data set (Piper et al., 2022). This model will/can be used
# to infer document vector representations for our annotated data, while training and testing our models.

import numpy as np
from gensim.models import LdaModel
from nltk import word_tokenize
from string import punctuation
import os
import random
from gensim.corpora import Dictionary

seed_value = 42  # random seed of 42 for all experiments
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


def main():
    documents = []
    count = 0
    NUM_TOPICS = 50
    for subdir, dirs, files in os.walk("../../dataset/minNarrative_txtfiles"):
        for file in files:
            count += 1
            filepath = os.path.join(subdir, file)
            with open(filepath, 'r') as f:
                text = f.read().strip()
            tokenized = word_tokenize(text)
            tokens = [word for word in tokenized if word not in punctuation]
            documents.append(tokens)
    train_x_dict = Dictionary(documents)
    train_corpus = [train_x_dict.doc2bow(text) for text in documents]
    print('Start training LDA model with {} documents'.format(count))
    lda = LdaModel(train_corpus, num_topics=NUM_TOPICS, random_state=42)
    print('Saving model')
    lda.save('lda/ldamodel')


if __name__ == '__main__':
    main()
