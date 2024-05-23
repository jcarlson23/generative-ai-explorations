#!/usr/bin/python3

import torch
import nltk 
import pdb 
nltk.download('punkt')

import math
import numpy as np 
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim 
from gensim.models import Word2Vec 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt 
import warnings 
warnings.filterwarnings( action = 'ignore')

sample = open('text.txt','r')
texts = sample.read()
contents = texts.replace('\n',' ')

data = list()

# parse the sentences
for i in sent_tokenize(contents):
    temp = list()
    # tokenize the sentece
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)

# create the skip gram model -2,1 SKIP 1,2
model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=512, window=5, sg=1)

# 1-The 2-black 3-cat 4-sat 5-on 6-the 7-couch 8-and 9-the 10-brown 11-dog 
# 12-slept 13-on 14-the 15-rug
word1 = 'black'
word2 = 'brown'
word3 = 'rug'
pos1 = 2
pos2 = 10
pos3 = 15

black_model = model2.wv[word1]
brown_model = model2.wv[word2]
rug_model = model2.wv[word3]

print(black_model)

# compute cosine similarities
black_brown_dot = np.dot(black_model,brown_model)
black_rug_dot = np.dot(black_model,rug_model)
brown_rug_dot = np.dot(brown_model,rug_model)

norm_black = np.linalg.norm(black_model)
norm_brown = np.linalg.norm(brown_model)
norm_rug = np.linalg.norm(rug_model)

cos_black_brown = black_brown_dot / (norm_black * norm_brown)
cos_black_rug = black_rug_dot / (norm_black * norm_rug)
cos_brown_rug = brown_rug_dot / (norm_brown * norm_rug)

print(f"Cosine black-brown {cos_black_brown}")
print(f"Cosine black-rug {cos_black_rug}")
print(f"Cosine brown-rug {cos_brown_rug}")

black_reshaped = black_model.reshape(1,512)
brown_reshaped = brown_model.reshape(1,512)
rug_reshaped = rug_model.reshape(1,512)

cos_brbl = cosine_similarity(black_reshaped, brown_reshaped)
cos_blco = cosine_similarity(black_reshaped, rug_reshaped)
cos_brco = cosine_similarity(brown_reshaped, rug_reshaped)

print(f"Cosine familiarity black-brown {cos_brbl}")
print(f"Cosine familiarity black-rug {cos_blco}")
print(f"Cosine familiarity brown-rug {cos_brco}")

pos_encoding1 = black_reshaped.copy()
pos_encoding2 = black_reshaped.copy()
pos_encoding3 = black_reshaped.copy()
pos_black     = black_reshaped.copy()
pos_brown     = brown_reshaped.copy() 
d = 512
max_length = 20 

for i in range(0, d, 2):
    pos_encoding1[0][i] = math.sin( pos1 / (10000 ** ((2*i)/d)))
    pos_black[0][i]    = (pos_black[0][i]*math.sqrt(d)) + pos_encoding1[0][i]
    pos_encoding1[0][i+1] = math.cos( pos1 / (10000 ** ((2*i)/d)))
    pos_black[0][i+1]    = (pos_black[0][i+1]*math.sqrt(d)) + pos_encoding1[0][i+1]
    # print out some diagnostic information
    print(i,pos_encoding1[0][i], i+1, pos_encoding1[0][i+1])
    print(i,pos_black[0][i],i+1,pos_black[0][i+1])
    print("\n")

max_len = d
pos_encoding = torch.zeros(d,d)
position = torch.arange(0,max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0,d,2).float() * (-math.log(10000.0) / d))
pos_encoding[:,0::2] = torch.sin(position * div_term)
pos_encoding[:,1::2] = torch.cos(position * div_term)

for i in range(0, d, 2):
    pos_encoding2[0][i] = math.sin( pos2 / (10000 ** ((2*i)/d)))
    pos_brown[0][i]    = (pos_brown[0][i]*math.sqrt(d)) + pos_encoding2[0][i]
    pos_encoding2[0][i+1] = math.cos( pos2 / (10000 ** ((2*i)/d)))
    pos_brown[0][i+1]    = (pos_brown[0][i+1]*math.sqrt(d)) + pos_encoding2[0][i+1]
    # print out some diagnostic information
    print(i,pos_encoding2[0][i], i+1, pos_encoding2[0][i+1])
    print(i,pos_brown[0][i],i+1,pos_brown[0][i+1])
    print("\n")

print(f"{word1} vs {word2}")
cos_ps  = cosine_similarity(pos_encoding1, pos_encoding2)
cos_pes = cosine_similarity(pos_black, pos_brown)
print(f"Word similarity: {cos_brbl}")
print(f"Positional similarity: {cos_ps}")
print(f"Positional encoding similarity {cos_pes}")

