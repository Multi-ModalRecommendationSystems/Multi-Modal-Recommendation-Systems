import numpy as np
import time


# Sculptor Optimization Algorithm (SOA)
def SOA(population, objective_function, lb, ub, max_iter):
    pop_size, dim = population.shape
    bounds = [lb, ub]
    best_solution = population[0]
    best_fitness = objective_function(best_solution)
    convergence= np.zeros(max_iter)
    CT = time.time()
    for t in range(max_iter):
        fitness = np.apply_along_axis(objective_function, 1, population)
        X_best = population[np.argmin(fitness)]  # Best in population

        for i in range(pop_size):
            beta = np.random.uniform()  # Random scalar
            # Phase 1: Equation (4)
            X_p1 = X_best + beta * (X_best - population[i])
            fitness_p1 = objective_function(X_p1)
            if fitness_p1 < fitness[i]:
                population[i] = X_p1

            # Phase 2: Equation (7)
            X_p2 = X_best - beta * (X_best - population[i])
            fitness_p2 = objective_function(X_p2)
            if fitness_p2 < fitness[i]:
                population[i] = X_p2

        # Update best solution
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = X_best
        convergence[t] = best_fitness
    ct = time.time() - CT

    return best_fitness, convergence, best_solution, ct
