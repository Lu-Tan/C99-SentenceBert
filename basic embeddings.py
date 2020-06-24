"""
2020/06/24
System: ubuntu 18.04
Python 3.6
Additional import: sentence bert

Function: Segment files into 3 segments and log boundary positions into file.
"""
from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging


def initialize():
    # Just some code to print debug information to stdout
    np.set_printoptions(threshold=100)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

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


def s(ranked_matrix, start, end):
    _sum = 0
    for i in range(start, end):
        for j in range(start, end):
            _sum += ranked_matrix[i, j]
    return _sum


def a(start, end):
    return (end-start + 1) ** 2


def D(ranked_matrix, end, i):
    sum_sk = s(ranked_matrix, 1, i) + s(ranked_matrix, i, end)
    sum_ak = a(1, i) + a(i + 1, end)
    temp = sum_sk / sum_ak
    return temp


def cluster(ranked_matrix, start, end):
    # divide into 2 pieces
    point_2 = start
    max = 0
    log = []
    for i in range(start+1, end-1):
        temp = D(ranked_matrix, end, i)
        log.append([i, temp])
        if temp > max:
            max = temp
            point_2 = i
    # print("half point is: " + str(point_2) + ", max value of D is " + str(max))
    # print("Log: " + str(log))
    return point_2  # return boundary


def log_result(result):
    file_result = open("file_result.txt", "a")
    file_result.write(str(result) + '\n')
    file_result.close()


def segmente_one_file(model, filename):
    # read sentences in a file and calculate its embeddings[]
    sentences = []
    with open(filename, 'r') as f:
        for line in f:
            sentence = "".join(line.strip('\n').split(','))  # Convert list to string
            sentence = sentence[3:]  # delete speaker and tab
            sentences.append(sentence)
    sentence_embeddings = model.encode(sentences)
    embeddings = []
    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(sentences, sentence_embeddings):
        embeddings.append(embedding)

    # calculate ranked_matrix from embeddings[]
    ranked_matrix = cal_rank(cal_cos_sim_matrix(embeddings), len(embeddings), 11)
    divide_into_2 = cluster(ranked_matrix, 0, len(embeddings))
    if divide_into_2 - 0 > len(embeddings) - divide_into_2:
        divide_into_3 = cluster(ranked_matrix, 0, divide_into_2)
    else:
        divide_into_3 = cluster(ranked_matrix, divide_into_2, len(embeddings))

    # log result[] as boundaries
    result = []
    if divide_into_2 > divide_into_3:
        result.append(divide_into_3)
        result.append(divide_into_2)
    else:
        result.append(divide_into_2)
        result.append(divide_into_3)
    # print("divide result is: " + str(result))

    # log boundaries of curret file into "file_result.txt"
    log_result(result)
    print("Boundary points of " + filename + " are " + str(result))


def main():
    initialize()
    # Load Sentence model (based on BERT) from URL
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    for i in range(1, 154):
        i = '%03d' % i
        filename = 'raw_conversation/' + 'high_' + str(i)
        # print(filename)
        segmente_one_file(model, filename)


main()

