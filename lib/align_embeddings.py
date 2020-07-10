"""
    Example of embedding's data structure
    embs = {
        a: [0,0,1,....,0],
        b: [1,0,0,....,0],
        ...
    }
"""

import numpy as np

def align_two_embeddings(emb_to_align, emb_base, common_keys=None):
    """
    :param emb_to_align: embedding vectors to be align
    :param emb_base: base embedding vectors
    :return:
        aligned_embeddings of emb_to_align
    """
    if not common_keys:
        common_keys = list(set(emb_to_align.keys()).intersection(set(emb_base.keys())))

    A = np.array([emb_to_align[key] for key in common_keys]).T
    B = np.array([emb_base[key] for key in common_keys]).T
    M = B.dot(A.T)
    u, sigma, v_t = np.linalg.svd(M)
    rotation_matrix = u.dot(v_t)
    aligned_embedding = {k: rotation_matrix.dot(v) for k, v in emb_to_align.items()}

    return aligned_embedding


def align_list_of_embeddingss(emb_list, emb_base):
    """
    :param emb_list: list of embedding vectors to be align
    :param emb_base: base embedding vectors
    :return:
        list of aligned_embeddings
    """
    common_keys = set.intersection(*[set(emb.keys()) for emb in emb_list])
    common_keys = list(common_keys.intersection(set(emb_base.keys())))

    aligned_embeddings = []
    for emb_to_align in emb_list:
        aligned_emb = align_two_embeddings(emb_to_align, emb_base, common_keys)
        aligned_embeddings.append(aligned_emb)

    return aligned_embeddings