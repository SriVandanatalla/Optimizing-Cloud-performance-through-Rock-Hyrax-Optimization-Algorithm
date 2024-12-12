import random
import numpy as np
import math
import matplotlib.pyplot as plt

# Function to read city names from an external file (cities.txt)
def load_cities_from_file(file_path):
    with open(file_path, 'r') as file:
        cities = [line.strip() for line in file.readlines()]
    return cities

# Generate random task distribution (cities assigned to servers)
def generate_random_task_distribution(num_tasks, num_servers):
    return [random.randint(1, num_servers) for _ in range(num_tasks)]

# Define heterogeneous server capacities (for example, different CPU speeds)
def generate_server_capacities(num_servers):
    return [random.uniform(1, 5) for _ in range(num_servers)]  # CPU capacities between 1 and 5

# Fitness function: Minimize the maximum load on any server, considering server capacities
def fitness_function(task_distribution, server_capacities):
    server_loads = [0] * len(server_capacities)
    
    for task in task_distribution:
        server_loads[task - 1] += 1 / server_capacities[task - 1]
    
    return max(server_loads)

# Mutation function: Randomly change the assignment of one task to a different server
def mutate(task_distribution, num_servers):
    idx = random.randint(0, len(task_distribution) - 1)
    task_distribution[idx] = random.randint(1, num_servers)  # Assign to a different server
    return task_distribution

# Crossover function: Combine two task distributions to create an offspring
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    offspring = parent1[:crossover_point] + parent2[crossover_point:]
    return offspring

# Rock Hyrax Optimization (RHO) for heterogeneous cloud server load balancing
def rock_hyrax_optimization(num_tasks, num_servers, server_capacities, population_size=100, generations=500, mutation_rate=0.1):
    population = [generate_random_task_distribution(num_tasks, num_servers) for _ in range(population_size)]
    best_solution = None
    best_fitness = float('inf')
    fitness_progress = []
    all_server_loads = []
    iteration_wise_distributions = []

    for generation in range(generations):
        fitness_scores = [fitness_function(individual, server_capacities) for individual in population]
        
        min_fitness = min(fitness_scores)
        min_index = fitness_scores.index(min_fitness)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[min_index]

        current_server_loads = [0] * num_servers
        for task in best_solution:
            current_server_loads[task - 1] += 1 / server_capacities[task - 1]
        all_server_loads.append(current_server_loads)
        iteration_wise_distributions.append(best_solution.copy())

        best_individuals = sorted(zip(population, fitness_scores), key=lambda x: x[1])[:population_size // 2]
        best_population = [individual for individual, _ in best_individuals]
        next_generation = best_population[:]

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(best_population, 2)
            offspring = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                offspring = mutate(offspring, num_servers)
            next_generation.append(offspring)

        population = next_generation
        fitness_progress.append(best_fitness)

    return best_solution, best_fitness, fitness_progress, all_server_loads, iteration_wise_distributions

# Visualization Functions
def plot_task_assignment(task_distribution, cities):
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(task_distribution)), task_distribution, c=task_distribution, cmap='viridis', marker='s')
    plt.colorbar(label='Server ID')
    plt.title("Task Assignment Visualization")
    plt.xlabel("Task Index (City)")
    plt.ylabel("Server Assigned")
    plt.show()

def plot_task_count_per_server(task_distribution, num_servers):
    task_counts = [task_distribution.count(server) for server in range(1, num_servers + 1)]
    plt.bar(range(1, num_servers + 1), task_counts, color='skyblue')
    plt.title("Task Count per Server")
    plt.xlabel("Server")
    plt.ylabel("Number of Tasks")
    plt.show()

def plot_fitness_vs_servers(fitness_progress):
    plt.plot(range(len(fitness_progress)), fitness_progress, color='orange', label='Fitness (Max Load)')
    plt.title("Fitness vs. Number of Servers")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

def plot_task_complexity_impact(server_capacities, fitness_progress):
    complexities = [1 / cap for cap in server_capacities]
    plt.scatter(complexities, [fitness_progress[-1]] * len(server_capacities), color='purple')
    plt.title("Task Complexity Impact")
    plt.xlabel("Complexity (1/Capacity)")
    plt.ylabel("Fitness")
    plt.show()

def plot_capacity_utilization(server_capacities, server_loads):
    utilization = [load / cap for load, cap in zip(server_loads, server_capacities)]
    plt.bar(range(1, len(server_capacities) + 1), utilization, color='pink')
    plt.title("Capacity Utilization per Server")
    plt.xlabel("Server")
    plt.ylabel("Utilization")
    plt.show()

def plot_load_imbalance_over_generations(all_server_loads):
    imbalances = [max(gen) - min(gen) for gen in all_server_loads]
    plt.plot(range(len(imbalances)), imbalances, color='red', label="Load Imbalance")
    plt.title("Load Imbalance Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Imbalance (Max - Min)")
    plt.legend()
    plt.show()

def plot_makespan_distribution(all_server_loads):
    makespans = [max(loads) for loads in all_server_loads]
    plt.hist(makespans, bins=20, color='green', edgecolor='black')
    plt.title("Makespan Distribution Across Servers")
    plt.xlabel("Makespan")
    plt.ylabel("Frequency")
    plt.show()

# Main
if __name__ == "__main__":
    cities = load_cities_from_file("cities.txt")
    num_tasks = len(cities)
    num_servers = 7  # Changed from 5 to 7
    server_capacities = generate_server_capacities(num_servers)

    best_solution, best_fitness, fitness_progress, all_server_loads, iteration_wise_distributions = rock_hyrax_optimization(num_tasks, num_servers, server_capacities)

    print("Best Solution (City Distribution to Servers):", best_solution)
    print("Best Fitness (Maximum Server Load):", best_fitness)

    plot_task_assignment(best_solution, cities)
    plot_task_count_per_server(best_solution, num_servers)
    plot_fitness_vs_servers(fitness_progress)
    plot_task_complexity_impact(server_capacities, fitness_progress)
    plot_capacity_utilization(server_capacities, all_server_loads[-1])
    plot_load_imbalance_over_generations(all_server_loads)
    plot_makespan_distribution(all_server_loads)
