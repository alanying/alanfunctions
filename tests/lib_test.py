# -*- coding: UTF-8 -*-

# Import from standard library
import os
import alanfunctions
import pandas as pd
# Import from our lib
from alanfunctions.lib import clean_data, embed_sentence, embedding
import pytest
import numpy as np
from gensim.models import Word2Vec

def test_clean_data():
    datapath = os.path.dirname(os.path.abspath(alanfunctions.__file__)) + '/data'
    df = pd.read_csv('{}/data.csv.gz'.format(datapath))
    first_cols = ['id', 'civility', 'birthdate', 'city', 'postal_code', 'vote_1']
    assert list(df.columns)[:6] == first_cols
    assert df.shape == (999, 142)
    out = clean_data(df)
    assert out.shape == (985, 119)

def test_embedded_sentence():
    sentences_train = ['id', 'civility', 'birthdate', 'city', 'postal_code', 'vote_1']
    example = ['this', 'movie', 'is', 'probably', 'the', 'worst', 'action', 'movie', 'ever']
    word2vec = Word2Vec(sentences=sentences_train)
    embedded_sentence = embed_sentence(word2vec, example)
    assert(type(embedded_sentence) == np.ndarray)


