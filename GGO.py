import numpy as np
import time


# Greylag Goose Optimization (GGO)
def GGO(population, obj_func, lb, ub, max_iter):
    num_agents, num_variables = population.shape
    fitness = np.zeros(num_agents)

    # Evaluate fitness of each agent
    for i in range(num_agents):
        fitness[i] = obj_func(population[i, :])
    convergence = np.zeros(max_iter)
    ct = time.time()
    # Main loop
    for iter in range(max_iter):
        # Update position and fitness of each agent
        for i in range(num_agents):
            # Determine the best agent in the flock
            best_agent_index = np.argmin(fitness)

            # Generate a new solution by combining exploration and exploitation
            new_solution = population[i, :] + np.random.rand(num_variables) * (
                    population[best_agent_index, :] - population[i, :])

            # Clip new solution to ensure it stays within bounds
            new_solution = np.clip(new_solution, lb[i], ub[i])

            # Evaluate fitness of the new solution
            new_fitness = obj_func(new_solution)

            # Update if the new solution is better
            if new_fitness < fitness[i]:
                population[i, :] = new_solution
                fitness[i] = new_fitness
        convergence[iter] = np.min(fitness)
    # Find the best solution in the final population
    best_fitness = np.min(fitness)
    best_index = np.argmin(fitness)
    best_solution = population[best_index, :]
    CT = time.time() - ct
    return best_fitness, convergence, best_solution, CT
