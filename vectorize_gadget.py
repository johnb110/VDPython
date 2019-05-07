import sys

import warnings
warnings.filterwarnings("ignore")

import pandas
from gensim.models import Word2Vec
import numpy


# DEBUG
#import joblib
#import pprint
#df = joblib.load('df.dump')
# ##

# Sets for operators
operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', 
    '!~', '<<', '>>', '<=', '>=', 
    '==', '!=', '&&', '||', '+=', 
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
    }
operators1 = { 
    '(', ')', '[', ']', '.', 
    '+', '-', '*', '&', '/', 
    '%', '<', '>', '^', '|', 
    '=', ',', '?', ':' , ';',
    '{', '}'
    }

"""
Takes a line of C++ code (string) as input
Tokenizes C++ code (breaks down into identifier, variables, keywords, operators)
Returns a list of tokens, preserving order in which they appear
"""
def tokenize(line):
    tmp, w = [], []
    i = 0
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i+3] in operators3:
            tmp.append(''.join(w))
            tmp.append(line[i:i+3])
            w = []
            i += 3
        elif line[i:i+2] in operators2:
            tmp.append(''.join(w))
            tmp.append(line[i:i+2])
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    # Filter out irrelevant strings
    res = list(filter(lambda c: c != '', tmp))
    return list(filter(lambda c: c != ' ', res))

"""
Uses Word2Vec to create a vector for each gadget
Treats lines in gadget as sentences
Creates model based on tokens in lines
Gets a vector for the gadget by averaging all token vectors
"""
def vectorize(gadget, vector_size=100):
    tokenized = []
    for line in gadget:
        tokenized.append(tokenize(line))
    model = Word2Vec(tokenized, min_count=1, size=vector_size, sg=1)
    embeddings = model.wv
    del model
    vectors = []
    size = 0
    for line in tokenized:
        for token in line:
            size += 1
            embedding = embeddings[token]
            if vectors == []:
                vectors = embedding
            else:
                vectors = numpy.add(vectors, embedding)
    return numpy.true_divide(vectors, size)
    

"""
Tokenizes every gadget in a pandas DataFrame
"""
def tokenize_df(df):
    tokenized_gadgets = []
    # Apply tokenize to every line of dataframe
    for line in range(len(df['gadget'])):
        l = list(map(tokenize, df["gadget"][line]))
        tokenized_gadgets.append(l)
    df['tokenized_gadget'] = tokenized_gadgets
    return df
