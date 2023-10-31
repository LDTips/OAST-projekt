import copy
import random
import math
from read_file import read_file


def random_sum(n, total):
    rand_floats = [random.random() for _ in range(n)]
    rand_nums = [math.floor(i * total / sum(rand_floats)) for i in rand_floats]
    for _ in range(total - sum(rand_nums)):
        rand_nums[random.randint(0, n-1)] += 1
    return rand_nums


def random_chromosome(demands):
    flows = {str(i): [] for i in range(1, len(demands) + 1)}
    for index, demand in enumerate(demands):
        h_d = demand.get('h_d')
        n = len(demand.get('paths'))
        flows[str(index + 1)] = random_sum(n, h_d)

    return flows


def calculate_link_loads(flows, demands, links):
    loads = {str(i): None for i in range(1, len(links) + 1)}
    flows_arr = []
    demands_arr = []
    for i in flows.values():
        flows_arr.extend(i)
    for i in demands:
        demands_arr.extend(i.get('paths'))

    for i in range(1, len(links)+1):
        indexes = [demands_arr.index(j) for j in demands_arr if i in j]
        load = sum(flows_arr[k] for k in indexes)
        loads[str(i)] = load

    return loads


def calculate_link_overloads(loads, links, module_cap):
    overloads = {str(i): None for i in range(1, len(links) + 1)}
    capacities = [module_cap * int(link.get('module_count')) for link in links]

    for index, (load, capacity) in enumerate(zip(loads.values(), capacities)):
        overloads[str(index + 1)] = int(load) - capacity

    return overloads


def crossover(parent1, parent2, crossover_probability):
    offspring1 = {str(i): [] for i in range(1, len(parent1) + 1)}
    offspring2 = {str(i): [] for i in range(1, len(parent1) + 1)}
    for i in offspring1:
        if random.random() <= crossover_probability:
            offspring1[i] = parent2[i]
            offspring2[i] = parent1[i]
        else:
            offspring1[i] = parent1[i]
            offspring2[i] = parent2[i]
    return offspring1, offspring2


def mutate(chromosomes, mutation_probability):
    for chromosome in chromosomes:
        for i in chromosome:
            if random.random() <= mutation_probability:
                decreased_index = random.randint(0, len(chromosome[i]) - 1)
                if chromosome[i][decreased_index] == 0:
                    continue
                increased_index = random.randint(0, len(chromosome[i]) - 1)
                while increased_index == decreased_index: # To shift one unit from one path to another indexes must be different
                    increased_index = random.randint(0, len(chromosome[i]) - 1)
                chromosome[i][decreased_index] -= 1
                chromosome[i][increased_index] += 1


def generate_start_population(population_size, demands):
    start_population = []
    while len(start_population) < population_size:
        start_population.append(random_chromosome(demands))
    return start_population


def find_best_objective_function(chromosomes, demands, links, module_capacity):
    best_objective_function = float('inf')
    best_chromosome = {}
    for chromosome in chromosomes:
        loads = calculate_link_loads(chromosome, demands, links)
        overloads = calculate_link_overloads(loads, links, module_capacity)
        objective_function = max(overloads.values())
        if objective_function < best_objective_function:
            best_objective_function = objective_function
            best_chromosome = chromosome
    return best_objective_function, best_chromosome


def find_worst_objective_function(chromosomes, demands, links, module_capacity):
    worst_objective_function = -float('inf')
    worst_chromosome = {}
    for chromosome in chromosomes:
        loads = calculate_link_loads(chromosome, demands, links)
        overloads = calculate_link_overloads(loads, links, module_capacity)
        objective_function = max(overloads.values())
        if objective_function > worst_objective_function:
            worst_objective_function = objective_function
            worst_chromosome = chromosome
    return worst_chromosome

# def old_main_loop(non_complex, demands, links):
#     crossover_probability = 0.5
#     mutation_probability = 0.1
#     limit = 50
#     min_f = float('inf')
#     iter_without_improvement = 0
#     while True:
#         flows = random_chromosome(demands)
#         loads = calculate_link_loads(flows, demands, links)
#         overloads = calculate_link_overloads(loads, links, non_complex.get('module_capacity'))
#
#         new_min = max(overloads.values())
#         if new_min < min_f:
#             min_f = new_min
#             best_overload = overloads
#             iter_without_improvement = 0
#
#         iter_without_improvement += 1
#         if iter_without_improvement >= limit:
#             return best_overload, min_f


def main_loop(non_complex, demands, links):
    population_size = 1000
    simulation_limit = 100
    crossover_occur_probability = 0.2
    crossover_probability = 0.5
    mutation_probability = 0.05
    iter_without_improvement = 0
    current_generation = 0
    parents_generation = generate_start_population(population_size, demands)
    best_solutions = []
    while True:
        next_generation = []
        current_generation += 1
        current_best_objective_function, current_best_chromosome = find_best_objective_function(parents_generation, demands, links, non_complex.get('module_capacity'))
        next_generation.append(copy.deepcopy(current_best_chromosome))
        best_solutions.append(copy.deepcopy(current_best_chromosome))

        # crossover criteria - crossover occur probability
        i = 0
        while i < len(parents_generation):
            if random.random() <= crossover_occur_probability:
                offspring1, offspring2 = crossover(parents_generation[i], parents_generation[i+1], crossover_probability)
                next_generation.append(offspring1)
                next_generation.append(offspring2)
            else:
                next_generation.append(parents_generation[i])
                next_generation.append(parents_generation[i+1])
            i += 2

        # we need to remove chromosome because we have "population_size+1" chromosome in next generation
        next_generation.remove(find_worst_objective_function(next_generation, demands, links, non_complex.get('module_capacity')))

        # mutation criteria - mutation probability - may be more than one mutation per chromosome
        mutate(next_generation, mutation_probability)

        next_generation_best_objective_function, next_generation_best_chromosome = find_best_objective_function(next_generation, demands, links, non_complex.get('module_capacity'))
        if next_generation_best_objective_function >= current_best_objective_function:
            iter_without_improvement += 1
        else:
            iter_without_improvement = 0

        parents_generation = next_generation

        if iter_without_improvement >= simulation_limit:
            return find_best_objective_function(best_solutions, demands, links, non_complex.get('module_capacity'))


def main():
    seed = 2077
    random.seed(seed)

    non_complex, demands, links = read_file('OPT-1 net4.txt')
    print(demands)
    print("Finished reading file")

    min_f, best_solution = main_loop(non_complex, demands, links)
    print("Simulation finished")
    print("Best solution:", best_solution)
    print("Objective function:", min_f)


if __name__ == "__main__":
    main()
