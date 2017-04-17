# 3/2017 | Moss McLaughlin


import sys
import csv
import itertools
import time
import operator
import io
import array
from datetime import datetime

import numpy as np
import nltk



SENTENCE_START_TOKEN = "sentence_start"
SENTENCE_END_TOKEN = "sentence_end"
UNKNOWN_TOKEN = "<UNK>"
ARTICLE_END_TOKEN = "</ARTICLE_END>"
NUM_TOKEN = "<NUM>"

similarity_threshold = 0.95
replace_similar_words = False

embedding_dim = 50
embedding_file = 'glove.6B/glove.6B.%sd.txt' % embedding_dim
data_file = 'data/train.csv'


def load_embeddings():
  print("Loading word embeddings...")
  with open(embedding_file) as f:
    words = {}
    w = [line.split(' ') for line in f]
    v = [x[1:] for x in w]
    w = [x[0] for x in w]
    print("Word embedding vocab size: ",len(v),'\n')
    

    for i in range(len(w)):
      words[w[i]] = v[i]

    return(words)



def create_embedding(embedding_dict,vocab_size,itw):
    print("Building word embedding matrix...")
    E = [None] * (vocab_size+3)
    for i in range(3): E[i] = np.zeros(embedding_dim) 
    for i in range(vocab_size-3): E[i+3] = embedding_dict[itw[i+3]]
    return E
        



def cos_similarity(x,y):
    x = np.array(x).astype(np.float)
    y = np.array(y).astype(np.float)
    d = np.dot(x,y) / (np.sqrt(np.dot(x,x))*(np.sqrt(np.dot(y,y))))
    return d



def find_similar_words(word,word_list,embeddings,similarity_threshold):
    similarity = [cos_similarity(embeddings[word],embeddings[w[0]]) for w in word_list]
    similarity = np.array(similarity)
    if similarity[similarity.argmax()] > similarity_threshold:
        return word,similarity.argmax()
    



def load_data(filename, vocabulary_size=2000):
    word_to_index = []
    index_to_word = []
    print("Reading text file...")
    with open(filename) as f:
        dat = csv.reader(f,delimiter=',',quotechar='"')
        #print('Raw training data:')
        #print(txt[0]) 
        #print('\n')
        #print(txt[-1])
        #print('\n')
        tokenized_sentences1 = [line[3] for line in dat]
        tokenized_sentences2 = [line[4] for line in dat]
        is_duplicate = [line[5] for line in dat]
        #tokenized_sentences1.pop(0)
        #tokenized_sentences2.pop(0)
        
    print("Parsed %d Questions.\n" % (len(tokenized_sentences1)*2))
    print("Tokenizing sentences...")
    tokenized_sentences1 = [nltk.word_tokenize(line) for line in tokenized_sentences1]
    tokenized_sentences2 = [nltk.word_tokenize(line) for line in tokenized_sentences2]
    print("Done.\n")

    print(tokenized_sentences1[0],tokenized_sentences1[-1])
    #tokenized_sentences1.pop(0)
    #tokenized_sentences2.pop(0)
    tokenized_sentences = [tokenized_sentences1,tokenized_sentences2] 
    txt = tokenized_sentences1
    for line in tokenized_sentences2: txt.append(line) 
    
    '''    
    # Replace all numbers with num_token
    for i,line in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w.isnumeric()==False else NUM_TOKEN for w in line]
    print("Done.\n")
    '''

    
    # Count word frequencies and build vocab
    word_freq = nltk.FreqDist(itertools.chain(*txt))
    n_data_words = len(word_freq.items())
    print("Found %d unique words tokens.\n" % n_data_words)

    
    embeddings = load_embeddings()
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    # Only consider words that are in GloVe embeddings.

    vocab = [w for w in vocab if w[0] in embeddings]
    n_glove_words = len(vocab)
    print("Found %d out of %d words in GloVe embeddings." % (n_glove_words,n_data_words))
    
    # We take the [vocabulary_size] most frequent words and build our word embedding matrix (or lookup table for now).  
    # Words in dataset are now either inside or outside embedding matrix.
    inside_words = sorted(vocab[:vocabulary_size], key=operator.itemgetter(1))
    outside_words = sorted(vocab[vocabulary_size:], key=operator.itemgetter(1))
    print("%d out of %d words appears more than once.\n" % (len(vocab),n_glove_words))
    
    index_to_word = ["<MASK/>", UNKNOWN_TOKEN,SENTENCE_END_TOKEN] + [x[0] for x in inside_words]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    E = create_embedding(embeddings,len(inside_words),index_to_word)
    
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocablary is '%s' and appeared %d times in this dataset.\n"           % (inside_words[1][0], inside_words[1][1]))
    
    # Find similar words that are in the data set but outside of our vocabulary
    if replace_similar_words:
        print("Searching for similar words...")
        similar_words = {}
        for w in outside_words:
            try: 
                similar_word,similar_index = find_similar_words(w[0],inside_words,embeddings,similarity_threshold)
                print("Replacing %s with %s" % (similar_word,inside_words[similar_index][0]))
                similar_words[similar_word] = inside_words[similar_index]
            except: None
            for line in tokenized_sentences:
                for x in line:
                    if x in similar_words: x = similar_words[x] 
                    
    
    # Save vocab in a file with one words in each line, from most to least frequent 
    #         (if same vocab is to be used for training and later evaluation)

   

    # Replace unknown words with unkown token (FOR NOW) 
    #  IN PROGRESS
    tokenized_sentences = [[[[w if w in word_to_index else UNKNOWN_TOKEN] for w in question] for question in channel] for channel in tokenized_sentences]

    print('Filtered training data:')
    print(tokenized_sentences1[1:5])
    print('\n')

    # Build training data
    X_train = np.asarray([[[word_to_index[w] for w in question[1:]] for question in channel] for channel in tokenized_sentences])  
    y_train = np.asarray(is_duplicate[1:])
    print(y_train[:10])
    return X_train,y_train,word_to_index,index_to_word

load_data(data_file)
