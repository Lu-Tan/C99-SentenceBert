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
import copy
import statistics


# import matplotlib.pyplot as plt


class Division:
    """
    a division by c99
    """
    Divisions = []
    Densitys = []
    Delta = []
    Matrix = None
    Length = 0
    Best_Division = 0

    def __init__(self, boundary_points, matrix=None):
        # boundary_points is a tuple

        if matrix is not None:
            self.initDivision()

        if Division.Matrix is None:
            Division.Matrix = matrix
            Division.Length = matrix.shape[0]
        self.boundary_points = boundary_points
        self.density = self.cal_density(boundary_points)

        Division.Divisions.append(self)
        Division.Densitys.append(self.density)

    def initDivision(self):
        Division.Divisions = []
        Division.Densitys = []
        Division.Delta = []
        Division.Matrix = None
        Division.Length = 0
        Division.Best_Division = 0

    def cal_density(self, boundary_points):
        sum_s = sum_a = 0
        for k in range(len(boundary_points) - 1):
            start, end = boundary_points[k], boundary_points[k + 1]
            for i in range(start, end):
                for j in range(start, end):
                    sum_s += Division.Matrix[i, j]
            sum_a = sum_a + (end - start) ** 2
        return sum_s / sum_a

    def find_next_division(self):
        new_d = -999
        new_bp = self.boundary_points

        for i in range(Division.Length):
            if i in self.boundary_points:
                # do nothing
                continue

            # add i into boundary_points
            bps = list(self.boundary_points)
            bps.append(i)
            bps.sort()
            temp_bp = tuple(bps)
            temp_d = self.cal_density(temp_bp)

            if temp_d > new_d:
                new_d = temp_d
                new_bp = temp_bp

        return Division(new_bp)

    def find_best_division(self):
        Ds = Division.Densitys
        delta = []
        for i in range(len(Ds) - 1):
            delta.append(Ds[i + 1] - Ds[i])
        Division.Delta = delta
        # # # smooth
        # mask = [1, 2, 4, 2, 1]
        #
        # delta = np.convolve(mask, delta, 'same')

        delta_delta = []
        for i in range(len(delta) - 1):
            delta_delta.append(delta[i + 1] - delta[i])
        b = delta_delta.index(min(delta_delta)) + 2
        Division.Best_Division = b
        return Division.Divisions[b]

    def save_to_file(self):
        # todo
        pass

    def __str__(self):
        # message for 'print'
        # print("boundary points : ", end='')
        # print(self.boundary_points)
        # print("all delta Densitys : ", end='')
        # print(Division.Delta)
        # print("best division times is %d" % Division.Best_Division)
        print(Division.Best_Division)
        print(Division.Length)
        print(Division.Best_Division / Division.Length)

        # plt.plot(Division.Densitys)
        return "division %d" % (len(self.boundary_points) - 1)


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
            sim_matrix[j, i] = sim_matrix[i, j]
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
            for x in range(int(i - (N - 1) / 2), int(i + (N - 1) / 2) + 1):
                for y in range(int(j - (N - 1) / 2), int(j + (N - 1) / 2) + 1):
                    if 0 <= x <= length - 1 and 0 <= y <= length - 1 and (x != i or y != j):
                        _num_examined += 1
                        if _value > matrix[x, y]:
                            _num_lower += 1
                        # print("i,j= " + str(i) + ", " + str(j) + " neighbour position: " + str(x) + ", " + str(y))
            ranked_matrix[i, j] = _num_lower / _num_examined
            # print("ranked_matrix[" + str(i) + "," + str(j) + "] = " + str(_num_lower) + "/" + str(_num_examined))
    return ranked_matrix


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
    matrix = ranked_matrix
    division = Division((0, matrix.shape[0]), matrix)
    for i in range(int(matrix.shape[0]/2)):
        division = division.find_next_division()
    best_division = division.find_best_division()
    print(best_division)
    # dD = dD[0:b]
    # u = statistics.mean(dD)
    # v = statistics.stdev(dD)
    # m = u + 1.2*v
    # divide_into_2 = cluster(ranked_matrix, 0, len(embeddings))
    # if divide_into_2 - 0 > len(embeddings) - divide_into_2:
    #     divide_into_3 = cluster(ranked_matrix, 0, divide_into_2)
    # else:
    #     divide_into_3 = cluster(ranked_matrix, divide_into_2, len(embeddings))
    #
    # # log result[] as boundaries
    # result = []
    # if divide_into_2 > divide_into_3:
    #     result.append(divide_into_3)
    #     result.append(divide_into_2)
    # else:
    #     result.append(divide_into_2)
    #     result.append(divide_into_3)
    # # print("divide result is: " + str(result))

    # log boundaries of curret file into "file_result.txt"
    log_result(best_division.boundary_points)
    # print("Boundary points of " + filename + " are " + str(result))


def main():
    initialize()
    # Load Sentence model (based on BERT) from URL
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    for i in range(1, 100):
        i = '%03d' % i
        filename = 'raw_conversation/' + 'high_' + str(i)
        # print(filename)
        segmente_one_file(model, filename)


main()
