import numpy as np
import pandas as pd
import random
import time
import itertools
from parser import parse_tsplib

# Calculate the Euclidean distance between two cities
def euclidean_distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))

# Calculate the total distance of a given route
def calculate_distance(route, coordinates):
    total_distance = sum(
        euclidean_distance(coordinates[route[i] - 1], coordinates[route[i + 1] - 1])
        for i in range(len(route) - 1)
    )
    total_distance += euclidean_distance(coordinates[route[-1] - 1], coordinates[route[0] - 1]) # Ensure we return to the start for the TSP
    return total_distance

# Initialize the population with random routes
def initialize_population(pop_size, num_cities):
    return [random.sample(range(1, num_cities + 1), num_cities) for _ in range(pop_size)]

# Select a parent using tournament selection
def tournament_selection(population, fitness, k):
    selected = random.sample(range(len(population)), k)
    return population[min(selected, key=lambda i: fitness[i])]

# Ordered crossover between two parents
def ordered_crossover(p1, p2):
    size = len(p1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = p1[start:end]
    p2_filtered = [city for city in p2 if city not in child]
    insert_index = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_filtered[insert_index]
            insert_index += 1
    return child

# Perform edge recombination crossover between two parents
def edge_crossover(p1, p2):
    size = len(p1)
    edge_table = {i: set() for i in p1}
    
    for parent in (p1, p2):
        for i in range(size):
            left = parent[i - 1]
            right = parent[(i + 1) % size]
            edge_table[parent[i]].update([left, right])
    
    child = []
    current = random.choice(p1)
    
    while len(child) < size:
        child.append(current)
        for edges in edge_table.values():
            edges.discard(current)
        
        next_city = None
        if edge_table[current]:
            next_city = min(edge_table[current], key=lambda x: len(edge_table[x]))
        
        if not next_city:
            remaining_cities = [city for city in p1 if city not in child]
            next_city = random.choice(remaining_cities) if remaining_cities else None
        
        current = next_city
    
    return child

# Perform swap mutation on a route
def swap_mutation(route):
    idx1, idx2 = random.sample(range(len(route)), 2)
    route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

# Perform reverse mutation on a route
def reverse_mutation(route):
    start, end = sorted(random.sample(range(len(route)), 2))
    route[start:end] = reversed(route[start:end])
    return route


def genetic_algorithm(tsp_data, pop_size, generations, crossover_rate, mutation_rate, k, crossover_type, mutation_type):
    start_time = time.time()
    num_cities = int(tsp_data['DIMENSION'])
    coordinates = tsp_data["NODE_COORDS"]
    population = initialize_population(pop_size, num_cities)
    best_solution, best_distance, best_route = None, float("inf"), None
    
    generation_thresholds = [100, 200, 300, 400, 500]
    threshold_fitness = {threshold: {'best': None, 'avg': None} for threshold in generation_thresholds}
    
    best_distances_over_time = []
    avg_distances_over_time = []
    
    for gen in range(generations):
        fitness = [calculate_distance(route, coordinates) for route in population]
        min_index = np.argmin(fitness)
        
        current_best = fitness[min_index]
        current_avg = sum(fitness) / len(fitness)
        
        best_distances_over_time.append(current_best)
        avg_distances_over_time.append(current_avg)

        if (gen + 1) in generation_thresholds:
            threshold_fitness[gen + 1]['best'] = current_best
            threshold_fitness[gen + 1]['avg'] = current_avg

        if current_best < best_distance:
            best_distance = current_best
            best_solution = population[min_index]
            best_route = best_solution.copy()

        new_population = []
        sorted_population = [x for _, x in sorted(zip(fitness, population))]
        elite_size = 2
        new_population.extend(sorted_population[:elite_size])

        # This allows us to switch between ordered and edge crossover for the gridsearch
        for _ in range((pop_size - elite_size) // 2):
            p1 = tournament_selection(population, fitness, k)
            p2 = tournament_selection(population, fitness, k)

            if crossover_type == 'ordered' and random.random() < crossover_rate:
                child1 = ordered_crossover(p1, p2)
                child2 = ordered_crossover(p2, p1)
            elif crossover_type == 'edg' and random.random() < crossover_rate:
                child1 = edge_crossover(p1, p2)
                child2 = edge_crossover(p2, p1)
            else:
                child1, child2 = p1[:], p2[:]

            if mutation_type == 'swap' and random.random() < mutation_rate:
                child1 = swap_mutation(child1)
                child2 = swap_mutation(child2)
            elif mutation_type == 'reverse' and random.random() < mutation_rate:
                child1 = reverse_mutation(child1)
                child2 = reverse_mutation(child2)

            new_population.extend([child1, child2])

        population = new_population

    runtime = time.time() - start_time
    return best_solution, best_distance, best_route, avg_distances_over_time, best_distances_over_time, runtime, threshold_fitness

# Perform grid search on the genetic algorithm and save results to CSV
def grid_search_ga_with_csv(tsp_data, pop_sizes, generations_list, crossover_rates, mutation_rates, k_values,crossover_types, mutation_types, output_file):
    results = []
    overall_best_route = None
    overall_best_distance = float('inf')
    
    total_combinations = len(list(itertools.product(
        pop_sizes, generations_list, crossover_rates, mutation_rates, k_values, crossover_types, mutation_types
    )))
    current_combination = 0
    
    for pop_size, generations, crossover_rate, mutation_rate, k, crossover_type, mutation_type in itertools.product(
        pop_sizes, generations_list, crossover_rates, mutation_rates, k_values, crossover_types, mutation_types
    ):
        current_combination += 1
        print(f"\nTesting combination {current_combination}/{total_combinations}")
        print(f"Parameters: pop_size={pop_size}, generations={generations}, crossover_rate={crossover_rate}, "
              f"mutation_rate={mutation_rate}, k={k}, crossover_type={crossover_type}, mutation_type={mutation_type}")
        
        best_solution, best_distance, best_route, avg_distances_over_time, best_distances_over_time, runtime, threshold_fitness = genetic_algorithm(
            tsp_data, pop_size, generations, crossover_rate, mutation_rate, k, crossover_type, mutation_type
        )
        
        if best_distance < overall_best_distance:
            overall_best_distance = best_distance
            overall_best_route = best_route
        
        result = {
            "pop_size": pop_size,
            "generations": generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "k": k,
            "crossover_type": crossover_type,
            "mutation_type": mutation_type,
            "best_distance": best_distance,
            "runtime": runtime
        }
        
        for gen, data in threshold_fitness.items():
            if data['best'] is not None:
                result[f"fitness_gen_{gen}_best"] = data['best']
                result[f"fitness_gen_{gen}_avg"] = data['avg']
        
        results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    best_result = min(results, key=lambda x: x["best_distance"])
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"\nBest Route Found: {' -> '.join(map(str, overall_best_route))} -> {overall_best_route[0]}")
    print(f"Best Distance: {overall_best_distance:.2f}")
    print("\nBest Configuration:")
    print(f"Population Size: {best_result['pop_size']}")
    print(f"Crossover Rate: {best_result['crossover_rate']}")
    print(f"Mutation Rate: {best_result['mutation_rate']}")
    print(f"Tournament Size (k): {best_result['k']}")
    print(f"Crossover Type: {best_result['crossover_type']}")
    print(f"Mutation Type: {best_result['mutation_type']}")
    print(f"Runtime: {best_result['runtime']:.2f} seconds")
    
    return results

if __name__ == "__main__":
    tsp_data = parse_tsplib("datasets/pr1002.tsp")
    # grid_search_results = grid_search_ga_with_csv(
    #     tsp_data,
    #     pop_sizes=[100,200,500],
    #     generations_list=[200],
    #     crossover_rates=[0.7,0.9],
    #     mutation_rates=[0.1,0.3],
    #     k_values=[3,5],
    #     crossover_types=['ordered', 'edg'],
    #     mutation_types=['swap', 'reverse'],
    #     output_file="berlin52results.csv"
    # )
    # grid_search_results = grid_search_ga_with_csv(
    #     tsp_data,
    #     pop_sizes=[50,100],
    #     generations_list=[500],
    #     crossover_rates=[0.7,0.9],
    #     mutation_rates=[0.1,0.3],
    #     k_values=[3,5],
    #     crossover_types=['ordered', 'edg'],
    #     mutation_types=['swap', 'reverse'],
    #     output_file="gapr1002_results.csv"
    # )
    grid_search_results = grid_search_ga_with_csv(
        tsp_data,
        pop_sizes=[100,],
        generations_list=[5000],
        crossover_rates=[0.7],
        mutation_rates=[0.3],
        k_values=[3],
        crossover_types=['edg'],
        mutation_types=['reverse',],
        output_file="finalpr_results.csv"
    )