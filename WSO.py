import numpy as np
import time


def initialization(soldiers_no, dim, ub, lb):
    return np.random.uniform(lb, ub, size=(soldiers_no, dim))


def WSO(positions, fobj, lb, ub, max_iter):
    # War Strategy Optimization Algorithm
    [soldiers_no, dim] = positions.shape
    king = np.zeros(dim)
    king_fit = np.inf
    positions = initialization(soldiers_no, dim, ub, lb)
    pop_size = positions.shape[0]
    convergence_curve = np.zeros(max_iter)
    positions_new = np.zeros_like(positions)
    fitness_old = np.full(pop_size, np.inf)
    fitness_new = np.full(pop_size, np.inf)
    l = 0  # Loop counter
    W1 = np.full(pop_size, 2)
    Wg = np.zeros(pop_size)
    trajectories = np.zeros((soldiers_no, max_iter))
    R = 0.1  # Select suitable value based on the application

    for j in range(pop_size):
        fitness = fobj(positions[j, :])
        fitness_old[j] = fitness
        if fitness < king_fit:
            king_fit = fitness
            king = positions[j, :]
    ct = time.time()
    while l < max_iter:
        idx = np.argsort(fitness_old)
        co = positions[idx[1], :]

        for i in range(pop_size):
            RR = np.random.rand()
            if RR < R:
                D_V = 2 * RR * (king - positions[i, :]) + 1 * W1[i] * np.random.rand() * (co - positions[i, :])
            else:
                D_V = 2 * RR * (co - king) + 1 * np.random.rand() * (W1[i] * king - positions[i, :])

            positions_new[i, :] = positions[i, :] + D_V
            positions_new[i, positions_new[i, :] > ub[i]] = ub[i, 0]
            positions_new[i, positions_new[i, :] < lb[i]] = lb[i, 0]
            fitness = fobj(positions_new[i, :])
            fitness_new[i] = fitness

            if fitness < king_fit:
                king_fit = fitness
                king = positions_new[i, :]

            if fitness < fitness_old[i]:
                positions[i, :] = positions_new[i, :]
                fitness_old[i] = fitness
                Wg[i] += 1
                W1[i] = 1 * W1[i] * (1 - Wg[i] / max_iter) ** 2

        if l < 1000:
            idx_max = np.argmax(fitness_old)
            positions[idx_max, :] = np.random.uniform(np.min(lb), np.max(ub), size=dim)
            positions[idx_max, :] = (1 - np.random.randn()) * (
                        king - np.median(positions[:pop_size, :], axis=0)) + positions[idx_max, :]
            positions[idx_max, :] = np.random.rand() * king
            W1[idx_max] = 0.5

        convergence_curve[l] = king_fit
        l += 1
    ct = time.time() - ct
    return king_fit, convergence_curve, king, ct
