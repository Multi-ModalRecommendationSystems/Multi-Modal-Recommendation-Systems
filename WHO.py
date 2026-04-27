import numpy as np
import time


def exchange(Stallion):
    NStallion = len(Stallion)
    a = np.random.permutation(NStallion)
    Stallion = [Stallion[i] for i in a]
    return Stallion


# Wild Horse Optimizer (WHO)
def WHO(X, fobj, lb, ub, Max_iter):
    [N, dim] = X.shape
    if len(ub) == 1:
        ub = np.ones(dim) * ub
        lb = np.ones(dim) * lb

    PS = 0.2  # Stallions Percentage
    PC = 0.13  # Crossover Percentage
    NStallion = int(np.ceil(PS * N))  # number Stallion
    Nfoal = N - NStallion
    Convergence_curve = np.zeros(Max_iter)
    gBest = np.zeros(dim)
    gBestScore = np.inf

    # create initial population
    empty = {'pos': None, 'cost': None, 'group': [None]*Nfoal}
    group = [empty.copy() for _ in range(Nfoal)]

    for i in range(Nfoal):
        group[i]['pos'] = lb + np.random.rand(dim) * (ub - lb)
        group[i]['cost'] = fobj(group[i]['pos'])

    Stallion = [empty.copy() for _ in range(NStallion)]

    for i in range(NStallion):
        Stallion[i]['pos'] = lb + np.random.rand(dim) * (ub - lb)
        Stallion[i]['cost'] = fobj(Stallion[i]['pos'])

    ngroup = len(group)
    a = np.random.permutation(ngroup)
    group = [group[i] for i in a]
    i = 0
    k = 1

    for j in range(ngroup):
        i += 1
        Stallion[i - 1]['group'][k - 1] = group[j]
        if i == NStallion:
            i = 0
            k += 1

    Stallion = exchange(Stallion)
    # value, index = min((s['cost'], i) for i, s in enumerate(Stallion))
    value, index = min(((np.min(s['cost']), i) for i, s in enumerate(Stallion)), key=lambda x: x[0])
    WH = Stallion[index]  # global
    gBest = WH['pos']
    gBestScore = WH['cost']
    Convergence_curve[0] = np.min(WH['cost'])
    ct = time.time()
    l = 0  # Loop counter
    while l < Max_iter:
        TDR = 1 - l * (1 / Max_iter)

        for o in range(NStallion):
            ngroup = len(Stallion[o]['group'])
            if Stallion[o]['group'] is not None and ngroup > 0:
                # _, index = zip(*sorted((s['cost'], l) for l, s in enumerate(Stallion[o]['group'])))
                _, index = zip(*sorted((np.sum(s['cost']), l) for l, s in enumerate(Stallion[o]['group']) if s is not None and 'cost' in s))
                Stallion[o]['group'] = [Stallion[o]['group'][j] for j in index]

            for j in range(4):
                if np.random.rand() > PC:
                    z = np.random.rand(dim) < TDR
                    r1, r2 = np.random.rand(), np.random.rand(dim)
                    idx = (z == 0)
                    r3 = r1 * idx + r2 * ~idx
                    rr = -2 + 4 * r3
                    Stallion[i]['group'][j]['pos'] = (
                            2 * r3 * np.cos(2 * np.pi * rr) * (Stallion[o]['pos'] - Stallion[o]['group'][j]['pos']) +
                            Stallion[o]['pos']
                    )
                else:
                    A = np.random.permutation(NStallion)
                    # A = A[A != i]
                    a, c = A[0], A[1]
                    x1 = Stallion[c]['group'][3]['pos']
                    x2 = Stallion[a]['group'][3]['pos']
                    y1 = (x1 + x2) / 2  # Crossover
                    Stallion[o]['group'][j]['pos'] = y1

                Stallion[o]['group'][j]['pos'] = np.minimum(Stallion[o]['group'][j]['pos'], ub)
                Stallion[o]['group'][j]['pos'] = np.maximum(Stallion[o]['group'][j]['pos'], lb)

                Stallion[o]['group'][j]['cost'] = fobj(Stallion[o]['group'][j]['pos'])

        R = np.random.rand()

        if R < 0.5:
            k = 2 * rr * np.cos(2 * np.pi * rr) * (WH['pos'] - Stallion[i]['pos']) + WH['pos']
        else:
            k = 2 * rr * np.cos(2 * np.pi * rr) * (WH['pos'] - Stallion[i]['pos']) - WH['pos']

        k = np.minimum(k, ub)
        k = np.maximum(k, lb)
        fk = fobj(k)
        for d in range(N):
            if fk[d] < Stallion[i]['cost'][d]:
                Stallion[i]['pos'] = k
                Stallion[i]['cost'] = fk

        Stallion = exchange(Stallion)

        # value, index = min((s['cost'], i) for i, s in enumerate(Stallion))
        value, index = min(((np.min(s['cost']), i) for i, s in enumerate(Stallion)), key=lambda x: x[0])
        if value < np.min(WH['cost']):
            WH = Stallion[index]

        gBest = WH['pos'][0]
        gBestScore = np.min(WH['cost'])
        Convergence_curve[l] = np.min(WH['cost'])
        l += 1
    ct = time.time() - ct
    return gBestScore, Convergence_curve, gBest, ct
