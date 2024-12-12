import numpy as np
import random
import matplotlib.pyplot as plt

# Define the objective function: Minimize server load imbalance
def objective_function(server_loads):
    # Metric to measure load imbalance (standard deviation of server loads)
    return np.std(server_loads)

# Initialize the population of solutions
def initialize_population(pop_size, num_servers, max_tasks):
    return [np.random.randint(0, max_tasks, num_servers) for _ in range(pop_size)]

# Evaluate the fitness of each solution
def evaluate_population(population):
    return [objective_function(individual) for individual in population]

# Update a solution based on RHO behavior
def update_solution(solution, best_solution, social_factor, escape_factor):
    new_solution = solution.copy()
    for i in range(len(solution)):
        if random.random() < social_factor:
            new_solution[i] += escape_factor * (best_solution[i] - solution[i])
        else:
            new_solution[i] += escape_factor * random.uniform(-1, 1)
    return np.clip(new_solution, 0, None).astype(int)

# Convert server numbers to city names for better readability
def convert_to_cities(solution, cities_mapping):
    return [cities_mapping[num] for num in solution]

# Read city names from an external file
def load_cities_from_file(file_path):
    with open(file_path, 'r') as file:
        cities = [line.strip() for line in file.readlines()]
    return cities

# Main RHO Algorithm
def rock_hyrax_optimization(
    num_servers, 
    max_tasks, 
    cities_mapping, 
    pop_size=30, 
    max_iterations=100, 
    social_factor=0.8, 
    escape_factor=0.5
):
    # Initialize population
    population = initialize_population(pop_size, num_servers, max_tasks)
    fitness = evaluate_population(population)
    best_solution = population[np.argmin(fitness)]
    fitness_over_time = [min(fitness)]
    print(f"Fitness Value: {fitness_over_time}")
    for iteration in range(max_iterations):
        new_population = []
        for individual in population:
            new_solution = update_solution(individual, best_solution, social_factor, escape_factor)
            new_population.append(new_solution)

        # Evaluate the new population
        population = new_population
        fitness = evaluate_population(population)
        best_solution = population[np.argmin(fitness)]
        fitness_over_time.append(min(fitness))

        # Logging progress
        print(f"Iteration {iteration + 1}, Best Fitness: {min(fitness)}")

    # Convert the best solution to cities for readability
    best_solution_cities = convert_to_cities(best_solution, cities_mapping)

    return best_solution, best_solution_cities, fitness_over_time

# Load city names from the external cities.txt file
cities_file = "cities.txt"  # Ensure this file exists in the same directory
city_names = load_cities_from_file(cities_file)

# Map server indices to city names dynamically
cities_mapping = {i: city_names[i] for i in range(len(city_names))}

# Parameters
num_servers = 10      # Number of servers (mapped to cities)
max_tasks = 100       # Max tasks any server can handle
pop_size = 50         # Population size
max_iterations = 200  # Number of iterations

# Run RHO
best_solution, best_solution_cities, fitness_over_time = rock_hyrax_optimization(
    num_servers, 
    max_tasks, 
    cities_mapping, 
    pop_size, 
    max_iterations
)

# Output results
print("Optimal Server Loads (Numeric):", best_solution)
print("Optimal Server Loads (Cities):", best_solution_cities)

# Plot fitness improvement over time

plt.plot(fitness_over_time)
plt.title("Fitness Improvement Over Time")
plt.xlabel("Iterations")
plt.ylabel("Best Fitness (Standard Deviation)")
plt.show()
