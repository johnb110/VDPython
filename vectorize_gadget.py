import re
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

class GadgetVectorizer:

    def __init__(self, vector_length):
        self.gadgets = []
        self.vector_length = vector_length
        self.forward_slices = 0
        self.backward_slices = 0

    """
    Takes a line of C++ code (string) as input
    Tokenizes C++ code (breaks down into identifier, variables, keywords, operators)
    Returns a list of tokens, preserving order in which they appear
    """
    @staticmethod
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

    @staticmethod
    def tokenize_gadget(gadget):
        tokenized = []
        function_regex = re.compile('FUN(\d)+')
        backwards_slice = False
        for line in gadget:
            tokens = GadgetVectorizer.tokenize(line)
            tokenized += tokens
            if len(list(filter(function_regex.match, tokens))) > 0:
                backwards_slice = True
            else:
                backwards_slice = False
        return tokenized, backwards_slice

    def add_gadget(self, gadget):
        tokenized_gadget, backwards_slice = GadgetVectorizer.tokenize_gadget(gadget)
        self.gadgets.append(tokenized_gadget)
        if backwards_slice:
            self.backward_slices += 1
        else:
            self.forward_slices += 1

    def vectorize(self, gadget):
        tokenized_gadget, backwards_slice = GadgetVectorizer.tokenize_gadget(gadget)
        vectors = numpy.zeros(shape=(50, self.vector_length))
        if backwards_slice:
            for i in range(min(len(tokenized_gadget), 50)):
                vectors[50 - 1 - i] = self.embeddings[tokenized_gadget[len(tokenized_gadget) - 1 - i]]
        else:
            for i in range(min(len(tokenized_gadget), 50)):
                vectors[i] = self.embeddings[tokenized_gadget[i]]
        return vectors

    def train_model(self):
        model = Word2Vec(self.gadgets, min_count=1, size=self.vector_length, sg=1)
        self.embeddings = model.wv
        del model
        del self.gadgets

    """
    Uses Word2Vec to create a vector for each gadget
    Treats lines in gadget as sentences
    Creates model based on tokens in lines
    Gets a vector for the gadget by averaging all token vectors
    """
    def _vectorize_old(self, gadget, vector_size=100):
        tokenized = []
        for line in gadget:
            tokenized.append(GadgetVectorizer.tokenize(line))
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
    
