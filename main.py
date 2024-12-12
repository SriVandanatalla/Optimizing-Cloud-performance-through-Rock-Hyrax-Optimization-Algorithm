import numpy as np
import random

def objective_function(server_loads):
    return np.std(server_loads)

def initialize_population(pop_size, num_servers, max_tasks):
    return [np.random.randint(0, max_tasks, num_servers) for _ in range(pop_size)]

def evaluate_population(population):
    return [objective_function(individual) for individual in population]

def update_solution(solution, best_solution, social_factor, escape_factor):
    new_solution = solution.copy()
    for i in range(len(solution)):
        if random.random() < social_factor:
            new_solution[i] += escape_factor * (best_solution[i] - solution[i])
        else:
            new_solution[i] += escape_factor * random.uniform(-1, 1)
    return np.clip(new_solution, 0, None).astype(int)

def rock_hyrax_optimization(
    num_servers, 
    max_tasks, 
    pop_size=30, 
    max_iterations=100, 
    social_factor=0.8, 
    escape_factor=0.5
):
    # Initialize population
    population = initialize_population(pop_size, num_servers, max_tasks)
    fitness = evaluate_population(population)
    best_solution = population[np.argmin(fitness)]

    for iteration in range(max_iterations):
        new_population = []
        for individual in population:
            new_solution = update_solution(individual, best_solution, social_factor, escape_factor)
            new_population.append(new_solution)

        # Evaluate the new population
        population = new_population
        fitness = evaluate_population(population)
        best_solution = population[np.argmin(fitness)]

        # Logging progress
        print(f"Iteration {iteration + 1}, Best Fitness: {min(fitness)}")

    return best_solution

# Parameters
num_servers = 10      # Number of servers
max_tasks = 100       # Max tasks any server can handle
pop_size = 50         # Population size
max_iterations = 200  # Number of iterations

# Run RHO
best_solution = rock_hyrax_optimization(num_servers, max_tasks, pop_size, max_iterations)

print("Optimal Server Loads:", best_solution)
