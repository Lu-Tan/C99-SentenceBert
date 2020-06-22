"""
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import sys


def cal_Euclidean_sim_matrix(embeddings):
    """
    Euclidean distance
    :param embeddings:
    :return:
    """
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


def cal_cos_sim_matrix(embeddings):
    """
    The similarity between two sentences is computed using the cosine measure.
    :param embeddings:
    :return:
    """
    _len = len(embeddings)
    # print("Count of embeddings " + str(_len))
    sim_matrix = np.zeros((_len, _len))
    for i in range(_len):
        # vec1 = np.array(embeddings[i])
        vec1 = np.mat(embeddings[i])
        for j in range(i, _len):
            # vec2 = np.array(embeddings[j])
            vec2 = np.mat(embeddings[j])
            # print("vec:" + str(vec2))
            # print("Length of vector: " + str(len(vec2)))
            num = float(vec1 * vec2.T)
            denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            cos = num / denom
            # distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
            sim_matrix[i, j] = 0.5 + 0.5 * cos
            sim_matrix[j, i] = sim_matrix[i, j]
    return sim_matrix

def cal_rank(matrix, length, N):
    ranked_matrix = np.zeros((length, length))
    # scan all values in the matrix
    for i in range(length):
        for j in range(length):
            _value = matrix[i, j]
            # print("i,j = " + str(i) + ", " + str(j))
            # count all legal neighbours
            _num_examined, _num_lower = 0, 0
            for x in range(int(i - (N-1) / 2), int(i + (N-1) / 2)+1):
                for y in range(int(j - (N-1)/2), int(j + (N-1)/2)+1):
                    if 0 <= x <= length-1 and 0 <= y <= length - 1 and ( x!=i or y!=j ):
                        _num_examined += 1
                        if _value > matrix[x, y]:
                            _num_lower += 1
                        # print("i,j= " + str(i) + ", " + str(j) + " neighbour position: " + str(x) + ", " + str(y))
            ranked_matrix[i, j] = _num_lower / _num_examined
            # print("ranked_matrix[" + str(i) + "," + str(j) + "] = " + str(_num_lower) + "/" + str(_num_examined))
    return ranked_matrix

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
# print("cal_sim_matrix(embeddings)")
# print(cal_cos_sim_matrix(embeddings))
print("ranked matrix: ")
print(cal_rank(cal_cos_sim_matrix(embeddings), len(embeddings), 11))




