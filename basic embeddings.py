"""
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import sys

def cal_sim_matrix(embeddings):
    _len = len(embeddings)
    sim_matrix = np.zeros((_len, _len))
    for i in range(_len):
        vec1 = np.array(embeddings[i])
        for j in range(i, _len):
            vec2 = np.array(embeddings[j])
            # distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
            sim_matrix[i, j] = np.sqrt(np.sum(np.square(vec1 - vec2)))
            sim_matrix[j,i] = sim_matrix[i,j]
    return sim_matrix

# Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
# /print debug information to stdout


# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('bert-base-nli-mean-tokens')

sentences = []
with open('high_001', 'r') as f:
    for line in f:
        sentence = "".join(line.strip('\n').split(','))  # Convert list to string
        sentence = sentence[3:]  # delete speaker and tab
        sentences.append(sentence)
# print(sentences)
# Embed a list of sentences
# sentences = ['This framework generates embeddings for each input sentence',
#              'Sentences are passed as a list of string.',
#              'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)

embeddings = []
# The result is a list of sentence embeddings as numpy arrays
for sentence, embedding in zip(sentences, sentence_embeddings):
    # print("Sentence:", sentence)
    # print("Embedding:", embedding)
    embeddings.append(embedding)
    print("")

# vec1 = np.array(embeddings[0])
# vec2 = np.array(embeddings[1])
# distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
# print("distance: " + str(distance))
print("cal_sim_matrix(embeddings)")
print(cal_sim_matrix(embeddings))



