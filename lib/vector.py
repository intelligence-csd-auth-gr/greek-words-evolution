import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity


def getCosineDistance(vectorA, vectorB):
    """
    Computes the cosine distance between two vectors A and B

    :param vectorA: the first vector to compare
    :param vectorB: the second vector to compare
    :return:
    """
    return spatial.distance.cosine(vectorA, vectorB)


def getCosineSimilarity(vectorA, vectorB):
    """
    Computes the cosine similarity between two vectors A and B

    :param vectorA: the first vector to compare
    :param vectorB: the second vector to compare
    :return:
        cosine_similarity
    """
    # assuming that the two vectors have already the same size
    size = len(vectorA)

    return cosine_similarity(vectorA.reshape(1, size), vectorB.reshape(1, size))[0][0]


def alignTwoEmbeddings(targetVector, baseVector, commonKeys=None):
    """
    Aligns two embeddings (essentially vectors)

    :param targetVector: embedding vectors to be align
    :param baseVector: base embedding vectors
    :param commonKeys:
    :return:
        alignedEmbedding of targetVector
    """
    """
        Example of embedding's data structure
        embs = {
            a: [0,0,1,....,0],
            b: [1,0,0,....,0],
            ...
        }
    """
    if not commonKeys:
        commonKeys = list(set(targetVector.keys()).intersection(set(baseVector.keys())))

    A = np.array([targetVector[key] for key in commonKeys]).T
    B = np.array([baseVector[key] for key in commonKeys]).T
    M = B.dot(A.T)
    u, sigma, v_t = np.linalg.svd(M)
    rotation_matrix = u.dot(v_t)
    alignedEmbedding = {k: rotation_matrix.dot(v) for k, v in targetVector.items()}

    return alignedEmbedding


def alignEmbeddingsList(targetVectors, baseVector):
    """
    :param targetVectors: list of embedding vectors to be align
    :param baseVector: base embedding vectors
    :return:
        list of alignedEmbeddings
    """
    commonKeys = set.intersection(*[set(emb.keys()) for emb in targetVectors])
    commonKeys = list(commonKeys.intersection(set(baseVector.keys())))

    alignedEmbeddings = []
    for targetVector in targetVectors:
        alignedEmbedding = alignTwoEmbeddings(targetVector, baseVector, commonKeys)
        alignedEmbeddings.append(alignedEmbedding)

    return alignedEmbeddings
