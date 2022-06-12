# This code creates a Doc2Vec model, using the complete data set. This model will/can be used
# to infer document vector representations for our annotated data, while training and testing our models.

import numpy as np
from nltk import word_tokenize
from string import punctuation
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import random

seed_value = 42  # random seed of 42 for all experiments
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


def main():
    documents = []
    count = 0
    for subdir, dirs, files in os.walk("../../dataset/minNarrative_txtfiles"):
        for file in files:
            count += 1
            filepath = os.path.join(subdir, file)
            with open(filepath, 'r') as f:
                text = f.read().strip()
            tokenized = word_tokenize(text)
            tokens = [word for word in tokenized if word not in punctuation]
            documents.append(TaggedDocument(tokens, [file]))

    print('Start training Doc2Vec model with {} documents'.format(count))
    doc2vec_model = Doc2Vec(documents, vector_size=200, workers=1, seed=42, epochs=20)
    print('Saving model')
    doc2vec_model.save('doc2vec/d2vmodel')


if __name__ == '__main__':
    main()
