import numpy as np


def get_features_for_one(NN_output, eps, k_list, n_classes, y_train):
    """Extrai o vetor de características KNN para uma única amostra.

    Recebe a saída de NearestNeighbors.kneighbors já calculada (distâncias e
    índices), de modo que apenas o cálculo numérico permaneça nesta função pura.
    """
    neighs = NN_output[1][0]
    neighs_dist = NN_output[0][0]
    neighs_y = y_train[neighs]

    return_list = []

    # Fração de objetos de cada classe.
    for k in k_list:
        bincount = np.bincount(neighs_y[:k])
        bincount = np.append(bincount, np.zeros(n_classes - len(bincount)))
        feats = []
        for i in range(n_classes):
            feats.append(bincount[i] / k)
        assert len(feats) == n_classes
        return_list += [feats]

    # Maior sequência de vizinhos com o mesmo rótulo do primeiro.
    a = np.where(neighs_y == neighs_y[0])[0]
    feats = [a[a == range(len(a))][-1] + 1]
    assert len(feats) == 1
    return_list += [feats]

    # Distância mínima até objetos de cada classe (999 se não houver).
    feats = []
    for c in range(n_classes):
        feats.append(np.min(np.where(neighs_y == c, neighs_dist, 999)))
    assert len(feats) == n_classes
    return_list += [feats]

    # Distância mínima normalizada até objetos de cada classe.
    feats = []
    closest_dist = np.min(neighs_dist)
    norm_dist = neighs_dist / (closest_dist + eps)
    for c in range(n_classes):
        dists = norm_dist[neighs_y == c]
        if len(dists) > 0:
            feats.append(dists.min())
        else:
            feats.append(999)
    assert len(feats) == n_classes
    return_list += [feats]

    # Distância ao k-ésimo vizinho e distância normalizada.
    for k in k_list:
        feat_51 = neighs_dist[k - 1]
        feat_52 = feat_51 / (neighs_dist[0] + eps)
        return_list += [[feat_51, feat_52]]

    # Distância média aos vizinhos de cada classe para cada k.
    for k in k_list:
        feats = []
        for c in range(n_classes):
            dists = neighs_dist[:k][neighs_y[:k] == c]
            if len(dists) > 0:
                feats.append(dists.sum() / (len(dists) + eps))
            else:
                feats.append(999)
        assert len(feats) == n_classes
        return_list += [feats]

    return np.hstack(return_list)
